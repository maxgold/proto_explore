from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch import jit
from dm_env import specs
import utils
from agent.ddpg_goal_gc import DDPGGoalGCAgent
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_replay_buffer, ndim_grid, make_replay_offline
import dmc
from logger import Logger, save

@jit.script
def sinkhorn_knopp(Q):
    Q -= Q.max()
    Q = torch.exp(Q).T
    Q /= Q.sum()

    r = torch.ones(Q.shape[0], device=Q.device) / Q.shape[0]
    c = torch.ones(Q.shape[1], device=Q.device) / Q.shape[1]
    for it in range(3):
        u = Q.sum(dim=1)
        u = r / u
        Q *= u.unsqueeze(dim=1)
        Q *= (c / Q.sum(dim=0)).unsqueeze(dim=0)
    Q = Q / Q.sum(dim=0, keepdim=True)
    return Q.T


class Projector(nn.Module):
    def __init__(self, pred_dim, proj_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(pred_dim, proj_dim), nn.ReLU(),
                                   nn.Linear(proj_dim, pred_dim))

        self.apply(utils.weight_init)

    def forward(self, x):
        return self.trunk(x)


class ProtoGoalGCAgent(DDPGGoalGCAgent):
    def __init__(self, pred_dim,proj_dim, queue_size, num_protos, tau,
                 encoder_target_tau, topk, update_encoder,update_gc, gc_only, 
                 offline, load_protos, task, frame_stack, action_repeat, replay_buffer_num_workers,
                 discount, reward_scores, reward_euclid, num_seed_frames, task_no_goal,
                 work_dir, goal_queue_size,**kwargs):
        super().__init__(**kwargs)
        self.first = True
        self.tau = tau
        self.encoder_target_tau = encoder_target_tau
        self.topk = topk
        self.num_protos = num_protos
        self.update_encoder = update_encoder
        self.update_gc = update_gc
        self.offline = offline
        self.gc_only = gc_only
        self.task = task
        self.frame_stack = frame_stack
        self.action_repeat = action_repeat
        self.replay_buffer_num_workers = replay_buffer_num_workers
        self.discount = discount
        self.reward_scores = reward_scores
        self.reward_euclid = reward_euclid
        self.num_seed_frames = num_seed_frames
        self.task_no_goal = task_no_goal
        self.work_dir = work_dir
        self.seed_until_step = utils.Until(self.num_seed_frames,
                                      self.action_repeat)
        self.goal_queue_size = goal_queue_size
        self.goal_topk = np.random.randint(0,100)
        self.load_protos = load_protos
        self.lr = .0001
        self.batch_size=256
        self.goal=None
        self.step=0
        self.reach_goal=False
        

        # models
        if self.gc_only==False:
            self.encoder_target = deepcopy(self.encoder)

            self.predictor = nn.Linear(self.obs_dim, pred_dim).to(self.device)
            self.predictor.apply(utils.weight_init)
            self.predictor_target = deepcopy(self.predictor)

            self.projector = Projector(pred_dim, proj_dim).to(self.device)
            self.projector.apply(utils.weight_init)

            # prototypes
            self.protos = nn.Linear(pred_dim, num_protos,
                                bias=False).to(self.device)
            self.protos.apply(utils.weight_init)
            #self.protos_target = deepcopy(self.protos)
            # candidate queue
            self.queue = torch.zeros(queue_size, pred_dim, device=self.device)
            self.queue_ptr = 0

            # optimizers
            self.proto_opt = torch.optim.Adam(utils.chain(
                self.encoder.parameters(), self.predictor.parameters(),
                self.projector.parameters(), self.protos.parameters()),
                                          lr=self.lr)

            self.predictor.train()
            self.projector.train()
            self.protos.train()
        
        elif self.load_protos:
            self.protos = nn.Linear(pred_dim, num_protos,bias=False).to(self.device)
            self.predictor = nn.Linear(self.obs_dim, pred_dim).to(self.device)
            self.projector = Projector(pred_dim, proj_dim).to(self.device)
          
        self.logger = Logger(self.work_dir,
                             use_tb=False,
                             use_wandb=True)

        self.goal_queue = torch.zeros(self.goal_queue_size, pred_dim, device=self.device)
        self.goal_queue_ptr = 0 
        
        idx = np.random.randint(0,400)
        goal_array = ndim_grid(2,20)
        self.first_goal = np.array([goal_array[idx][0], goal_array[idx][1]])
        self.train_env1 = dmc.make(self.task_no_goal, self.obs_type, self.frame_stack,
                                   self.action_repeat, seed=None, goal=self.first_goal)
        
        # get meta specs
        self.meta_specs = self.get_meta_specs()
        # create replay buffer
        self.data_specs = (self.train_env1.observation_spec(),
                      self.train_env1.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage1 = ReplayBufferStorage(self.data_specs, self.meta_specs,
                                                  self.work_dir / 'buffer1')

        # create replay buffer
        self.replay_loader1 = make_replay_loader(self.replay_storage1,
                                                False,
                                                1000000,
                                                self.batch_size,
                                                self.replay_buffer_num_workers,
                                                False, 3, self.discount,
                                                True, False,self.obs_type, goal_proto=True)
        
        self._replay_iter1 = None


    @property
    def replay_iter1(self):
        if self._replay_iter1 is None:
            self._replay_iter1 = iter(self.replay_loader1)
        return self._replay_iter1
    
    @property
    def replay_iter2(self):
        if self._replay_iter2 is None:
            self._replay_iter2 = iter(self.replay_loader2)
        return self._replay_iter2


    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        utils.hard_update_params(other.predictor, self.predictor)
        utils.hard_update_params(other.projector, self.projector)
        utils.hard_update_params(other.protos, self.protos)
        if self.init_critic:
            utils.hard_update_params(other.critic, self.critic)

    def init_encoder_from(self, encoder):
        utils.hard_update_params(encoder, self.encoder)
        
    def init_protos_from(self, protos):
        utils.hard_update_params(protos.protos, self.protos)

    def normalize_protos(self):
        C = self.protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self.protos.weight.data.copy_(C)

    def compute_intr_reward(self, obs, step):
        self.normalize_protos()
        # find a candidate for each prototype
        with torch.no_grad():
            z = self.encoder(obs)
            z = self.predictor(z)
            z = F.normalize(z, dim=1, p=2)
            scores = self.protos(z).T
            prob = F.softmax(scores, dim=1)
            candidates = pyd.Categorical(prob).sample()

        #if step>300000:
        #    import IPython as ipy; ipy.embed(colors='neutral')
        # enqueue candidates
        ptr = self.queue_ptr
        self.queue[ptr:ptr + self.num_protos] = z[candidates]
        self.queue_ptr = (ptr + self.num_protos) % self.queue.shape[0]

        # compute distances between the batch and the queue of candidates
        z_to_q = torch.norm(z[:, None, :] - self.queue[None, :, :], dim=2, p=2)
        all_dists, _ = torch.topk(z_to_q, self.topk, dim=1, largest=False)
        dist = all_dists[:, -1:]
        reward = dist
        return reward

    def update_proto(self, obs, next_obs, step):
        metrics = dict()

        # normalize prototypes
        self.normalize_protos()

        # online network
        s = self.encoder(obs)
        s = self.predictor(s)
        s = self.projector(s)
        s = F.normalize(s, dim=1, p=2)
        scores_s = self.protos(s)
        #import IPython as ipy; ipy.embed(colors='neutral')
        log_p_s = F.log_softmax(scores_s / self.tau, dim=1)

        # target network
        with torch.no_grad():
            t = self.encoder_target(next_obs)
            t = self.predictor_target(t)
            t = F.normalize(t, dim=1, p=2)
            scores_t = self.protos(t)
            q_t = sinkhorn_knopp(scores_t / self.tau)

        # loss
        loss = -(q_t * log_p_s).sum(dim=1).mean()
        if self.use_tb or self.use_wandb:
            metrics['repr_loss'] = loss.item()
        self.proto_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.proto_opt.step()

        return metrics
    
    def roll_out(self, global_step):
        if self.first:
            self.first=False
            self.episode_step, self.episode_reward = 0, 0
            self.time_step1 = self.train_env1.reset()
            self.meta = self.init_meta()
            
            protos = self.protos.weight.data.detach().clone()
            self.goal=protos[0][None,:]
            ptr = self.goal_queue_ptr
            self.goal_queue[ptr] = self.goal
            self.goal_queue_ptr = (ptr + 1) % self.goal_queue.shape[0]

            with torch.no_grad():
                obs = self.time_step1.observation['pixels']
                obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                #import IPython as ipy; ipy.embed(colors='neutral')
                self.z = self.encoder(obs)
                self.z = self.predictor(self.z)
                self.z = self.projector(self.z)
                self.z = F.normalize(self.z, dim=1, p=2)
#                 scores_z = self.protos(self.z)
                self.reward =0
                
            self.replay_storage1.add_proto_goal(self.time_step1, self.z.cpu().numpy(), self.meta, self.goal.cpu().numpy(), self.reward, last=False)
            
        else:
            
            #no reward for too  long so sample goal nearby 
            if (self.episode_step == 500 and self.reach_goal==False) or global_step==0:
                self.episode_step=0
                self.step=0
                #sample proto 
                #not finished yet 
                #sample a prototype close to current obs.
                self.replay_storage1.add_proto_goal(self.time_step1, self.z.cpu().numpy(), self.meta, self.goal.cpu().numpy(), self.reward.cpu().numpy(), last=True)
                self.time_step1 = self.train_env1.reset()
                protos = self.protos.weight.data.detach().clone()
                with torch.no_grad():
                    obs = self.time_step1.observation['pixels']
                    obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                    self.z = self.encoder(obs)
                    self.z = self.predictor(self.z)
                    self.z = self.projector(self.z)
                    self.z = F.normalize(self.z, dim=1, p=2)
#                     scores_z = self.protos(self.z)
                    self.reward =0
                self.gaol_topk = np.random.randint(0,10)
                z_to_proto = torch.norm(self.z[:, None, :] - protos[None, :, :], dim=2, p=2)
                all_dists, _ = torch.topk(z_to_proto, self.goal_topk, dim=1, largest=False)
                idx = _[:,-1]
                
                self.goal = protos[idx]
                ptr = self.goal_queue_ptr
                self.goal_queue[ptr] = self.goal
                self.goal_queue_ptr = (ptr + 1) % self.goal_queue.shape[0]
                
            meta = self.update_meta(self.meta, global_step, self.time_step1)
            
            # sample action
            with torch.no_grad():
                action1 = self.act(self.z, 
                                   self.goal, 
                                   self.meta, 
                                   global_step, 
                                   eval_mode=False)
            
            # take env step
            self.time_step1 = self.train_env1.step(action1)
            
            with torch.no_grad():
                obs = self.time_step1.observation['pixels']
                obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                self.z = self.encoder(obs)
                self.z = self.predictor(self.z)
                self.z = self.projector(self.z)
                self.z = F.normalize(self.z, dim=1, p=2)
                scores_z = self.protos(self.z)
            
            if self.reward_scores:
                self.reward = scores_z
                print('reward_scores', self.reward)
            elif self.reward_euclid:
                self.reward = -torch.norm(self.z[:, None, :] - self.goal[None, :,:], dim=2, p=2)
                print('reward_euclid', self.reward)
            #if max score == self.goal then reward = 1, otherwise =0
            
            self.episode_reward += self.reward 
            self.episode_step += 1
            self.step +=1

            if self.step!=500:
                self.replay_storage1.add_proto_goal(self.time_step1,self.z.cpu().numpy(), self.meta, self.goal.cpu().numpy(), self.reward.cpu().numpy(), False)
            elif self.step!=self.episode_step and self.step==500:
                self.step=0
                self.replay_storage1.add_proto_goal(self.time_step1,self.z.cpu().numpy(), self.meta, self.goal.cpu().numpy(), self.reward.cpu().numpy(), last=True)
             #    self.replay_storage2.add(time_step2, meta, True)
            
            
            
            idx = None
            if (self.reward_scores and self.episode_reward > 100) or (self.reward_euclid and self.reward>-.1):
                self.reach_goal=True
                self.episode_step, self.episode_reward = 0, 0
                #import IPython as ipy; ipy.embed(colors='neutral')   
                protos = self.protos.weight.data.detach().clone()
                protos_to_goal_queue = torch.norm(protos[:, None,:] - self.goal_queue[None, :, :], dim=2, p=2)
                #512xself.goal_queue.shape[0]
                all_dists, _ = torch.topk(protos_to_goal_queue, self.goal_queue.shape[0]*5, dim=2, largest=False)
                #criteria of protos[idx] should be: 
                #current self.goal's neghbor but increasingly further away from previous golas
                for x in range(5,self.goal_queue.shape[0]*5):
                    #select proto that's 5th closest to current goal 
                    #current goal is indexed at self.goal_queue_ptr-1
                    idx = _[x, self.goal_queue_ptr-1]
                    for y in range(1,self.goal_queue.shape[0]):
                        if idx.isin(_[:5*(self.goal_queue.shape[0]-y), (self.goal_queue_ptr-1+y)%self.goal_queue.shape[0]]):
                            continue
                        else:
                            break
                    break
                
                if idx is None:
                    print('nothing fits sampling criteria, reevaluate')
                    idx = np.random.randint(0,protos.shape[0])
                    self.goal=protos[idx]
                           
                self.goal = protos[idx]
                #resample proto goal based on most recent k number of sampled goals(far from them) but close to current observaiton

            if not self.seed_until_step(global_step):
                metrics = self.update(self.replay_iter1, global_step, actor1=True)
#                 metrics = self.agent.update(self.replay_iter2, self.global_step)
                self.logger.log_metrics(metrics, global_step*2, ty='train')
            
            
    def update(self, replay_iter, step, actor1=False):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        if actor1 and step % self.update_gc==0:
            obs, action, extr_reward, discount, next_obs, goal = utils.to_torch(
            batch, self.device)
            if self.obs_type=='states':
                goal = goal.reshape(-1, 2).float()
            
        elif actor1==False:
            obs, action, extr_reward, discount, next_obs = utils.to_torch(
                    batch, self.device)
        else:
            return metrics
        
        action = action.reshape(-1,2)
        discount = discount.reshape(-1,1)
        extr_reward = extr_reward.float()

        # augment and encode
       # with torch.no_grad():
       #     obs = self.aug(obs)
       #     next_obs = self.aug(next_obs)
       #     if actor1:
       #         goal = self.aug(goal)
           
        if actor1==False:

            if self.reward_free and step<300000:
                metrics.update(self.update_proto(obs, next_obs, step))
                with torch.no_grad():
                    intr_reward = self.compute_intr_reward(next_obs, step)

                if self.use_tb or self.use_wandb:
                    metrics['intr_reward'] = intr_reward.mean().item()
                
                reward = intr_reward
            else:
                reward = extr_reward

            if self.use_tb or self.use_wandb:
                metrics['extr_reward'] = extr_reward.mean().item()
                metrics['batch_reward'] = reward.mean().item()

            obs = self.encoder(obs)
            next_obs = self.encoder(next_obs)

            if not self.update_encoder:
                obs = obs.detach()
                next_obs = next_obs.detach()
            # update critic
            metrics.update(
                self.update_critic2(obs.detach(), action, reward, discount,
                               next_obs.detach(), step))

            # update actor
            metrics.update(self.update_actor2(obs.detach(), step))

            # update critic target
            #if step <300000:

            utils.soft_update_params(self.predictor, self.predictor_target,
                                 self.encoder_target_tau)
            utils.soft_update_params(self.encoder, self.encoder_target,
                                                    self.encoder_target_tau)
            utils.soft_update_params(self.critic2, self.critic2_target,
                                 self.critic2_target_tau)

        elif actor1 and step % self.update_gc==0:
            reward = extr_reward
            if self.use_tb or self.use_wandb:
                metrics['extr_reward'] = extr_reward.mean().item()
                metrics['batch_reward'] = reward.mean().item()

       #     obs = self.encoder(obs)
        #    next_obs = self.encoder(next_obs)
            #goal = self.encoder(goal)

          #  if not self.update_encoder:
            
            obs = obs.detach()
            next_obs = next_obs.detach()
             #   goal=goal.detach()
        
            # update critic
            metrics.update(
                self.update_critic(obs.detach(), goal, action, reward, discount,
                               next_obs.detach(), step))
            # update actor
            metrics.update(self.update_actor(obs.detach(), goal, step))

            # update critic target
            utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
    

    def get_q_value(self, obs,action):
        Q1, Q2 = self.critic2(torch.tensor(obs).cuda(), torch.tensor(action).cuda())
        Q = torch.min(Q1, Q2)
        return Q

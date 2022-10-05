import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import seaborn as sns; sns.set_theme()
import pandas as pd
from copy import deepcopy
from logger import Logger, save
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
import matplotlib.pyplot as plt
from video import TrainVideoRecorder, VideoRecorder
import os
import wandb
from pathlib import Path
from collections import defaultdict
import io

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


class ProtoGoalGCGridAgent(DDPGGoalGCAgent):
    def __init__(self, pred_dim,proj_dim, queue_size, num_protos, tau,
                 encoder_target_tau, topk, update_encoder,update_gc, gc_only, 
                 offline, load_protos, task, frame_stack, action_repeat, replay_buffer_num_workers,
                 discount, reward_nn, reward_scores, reward_scores_dense, reward_euclid,
                 pos_reward, neg_reward, num_seed_frames, task_no_goal, work_dir, goal_queue_size, 
                 tmux_session, eval_every_frames, seed, eval_after_step, episode_length, hybrid_gc, 
                 hybrid_pct, batch_size_gc, const_init, episode_reset_length, use_closest_proto, 
                 **kwargs):
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
        self.reward_nn = reward_nn
        self.reward_scores = reward_scores
        self.reward_scores_dense = reward_scores_dense
        self.reward_euclid = reward_euclid
        self.pos_reward = pos_reward
        self.neg_reward = neg_reward
        self.num_seed_frames = num_seed_frames
        self.task_no_goal = task_no_goal
        self.work_dir = work_dir
        self.seed_until_step = utils.Until(self.num_seed_frames,self.action_repeat)
        self.goal_queue_size = goal_queue_size
        self.goal_topk = np.random.randint(3,10)
        self.load_protos = load_protos
        self.tmux_session = tmux_session
        self.eval_every_frames=eval_every_frames
        self.eval_every_step = utils.Every(self.eval_every_frames, self.action_repeat)
        self.seed = seed
        self.eval_after_step = eval_after_step
        self.episode_length = episode_length
        self.hybrid_gc = hybrid_gc
        self.hybrid_pct = hybrid_pct
#         self.lr = .0001
        self.batch_size=256
        self.batch_size_gc=batch_size_gc
        self.goal=None
        self.step=0
        self._global_episode=0
        self.constant_init_dist = False
        self.const_init = const_init
        self.episode_reset_length = episode_reset_length
        self.use_closest_proto = use_closest_proto
        print('lr', self.lr)

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
        
        self.logger = Logger(self.work_dir, use_tb=self.use_tb, use_wandb=self.use_wandb)
        work_path = str(os.getcwd().split('/')[-2])+'/'+str(os.getcwd().split('/')[-1])
        exp_name = '_'.join([
                'exp', 'proto_goal_gc', 'pmm', self.obs_type, str(self.seed), str(self.tmux_session),work_path
            ])
        wandb.init(project="urlb", group='proto_goal_gc', name=exp_name) 

        self._global_episode = 0

        self.goal_queue = torch.zeros(self.goal_queue_size, pred_dim, device=self.device)
        self.goal_queue_ptr = 0 
        self.ten_step_reward = torch.zeros(10, 1, device=self.device)
        self.ten_step_queue_ptr = 0
        self.constant_init_env = False
        self.ts_init = None
        self.z = None
        self.obs2 = None
        self.goal_key = None
        self.state_proto_pair = defaultdict(list)
        self.goal_freq = torch.zeros((self.num_protos,))
        self.count=0
        
        goal_array = ndim_grid(2,20)
        idx = np.random.randint(goal_array.shape[0])
        self.first_goal = np.array([goal_array[idx][0], goal_array[idx][1]])
        self.train_env1 = dmc.make(self.task_no_goal, self.obs_type, self.frame_stack,
                                   self.action_repeat, seed=None, goal=self.first_goal)

        self.eval_env = dmc.make(self.task_no_goal, self.obs_type, self.frame_stack,
                                                   self.action_repeat, seed=None, goal=self.first_goal)
        
        self.eval_env_goal = dmc.make(self.task_no_goal, self.obs_type, self.frame_stack,
                                                   self.action_repeat, seed=None, goal=self.first_goal)
        lst =[]
        for ix,x in enumerate(goal_array):
            print(x[0])
            print(x[1])
            if (-.2<x[0]<.2 and -.02<x[1]<.02) or (-.02<x[0]<.02 and -.2<x[1]<.2):
                lst.append(ix)
                print('del',x)
        self.goal_array=np.delete(goal_array, lst,0)
        self.goal_array_loaded=False
        self.goal_tensor = torch.as_tensor(self.goal_array)
        self.previous_goal = None

        init_state = [-.15, .15]
        with torch.no_grad():
            with self.eval_env_goal.physics.reset_context():
                self.time_step_init = self.eval_env_goal.physics.set_state(np.array([init_state[0], init_state[1], 0, 0]))
                
            self.time_step_init = self.eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
            
            self.time_step_init = np.transpose(self.time_step_init, (2,0,1))
        
        self.proto_goal = []
        self.encoded_goal = []
        self.grid_proto_goal = []
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

        if self.hybrid_gc:
            self.replay_loader1 = make_replay_loader(self.replay_storage1,
                                                False,
                                                1000000,
                                                self.batch_size_gc,
                                                self.replay_buffer_num_workers,
                                                False, 1, self.discount,
                                                True, hybrid=self.hybrid_gc,obs_type=self.obs_type, 
						hybrid_pct=self.hybrid_pct, goal_proto=True, agent=self) 
        
        else:
            self.replay_loader1 = make_replay_loader(self.replay_storage1,
                                                False,
                                                1000000,
                                                self.batch_size_gc,
                                                self.replay_buffer_num_workers,
                                                False, 3, self.discount,
                                                True, False,self.obs_type,
                                                hybrid_pct=5,goal_proto=True,agent=self)
        
        self._replay_iter1 = None
        self.timer = utils.Timer()
         
        self.video_recorder = VideoRecorder(
                                        self.work_dir,
                                        camera_id=0,
                                        use_wandb=True)
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
    
    def roll_out(self, global_step, curriculum):
        file_name=self.work_dir / './state_proto_similarity_{global_step}.npz'
        if global_step%100000==0:
            with io.BytesIO() as bs:
                np.savez_compressed(bs, **self.state_proto_pair)
                bs.seek(0)
                with file_name.open("wb") as f:
                    f.write(bs.read())

        tmp_encoded_goal = defaultdict(list)
        tmp_proto_goal = defaultdict(list)
        protos = self.protos.weight.data.detach().clone()

        if self.goal_array_loaded==False:
            for x in self.goal_array:
                #self.state_2 = torch.as_tensor(x)
                state = np.array([x[0], x[1], 0, 0])
                with torch.no_grad():
                    with self.eval_env_goal.physics.reset_context():
                        time_step_init = self.eval_env_goal.physics.set_state(np.array(state))
                    time_step_init = self.eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
                    time_step_init = np.transpose(time_step_init, (2,0,1))
                    time_step_init = np.tile(time_step_init, (3,1,1))

                    obs = time_step_init
                    obs = torch.as_tensor(obs.copy(), device=self.device).unsqueeze(0)
                    z = self.encoder(obs)
                    tmp_encoded_goal['state'].append(state)
                    tmp_encoded_goal['observation'].append(z.detach().clone().cpu().numpy())
                    self.encoded_goal.append(z)
                    z = self.predictor(z)
                    z = self.projector(z)
                    z = F.normalize(z, dim=1, p=2)
                    self.grid_proto_goal.append(z)
                    #z_to_proto = torch.norm(z[:, None, :] - protos[None, :, :], dim=2, p=2)
                    #all_dists, _ = torch.topk(z_to_proto, 1, dim=1, largest=False)
                    sim = self.protos(z)
                    idx = sim.argmax()
                    if self.use_closest_proto==False:
                        self.proto_goal.append(z)
                    else:
                        self.proto_goal.append(protos[idx][None,:])
                        tmp_proto_goal['closest_proto'].append(protos[idx][None,:].detach().clone().cpu().numpy())
                        
                    tmp_proto_goal['state'].append(state)
                    tmp_proto_goal['observation'].append(z.detach().clone().cpu().numpy())
                    

            fn = [Path('./encoded_grid_goals.npz'), Path('./proto_grid_goals.npz')]
            lst_encoded = [tmp_encoded_goal, tmp_proto_goal]
            self.encoded_goal = torch.cat(self.encoded_goal, axis=0)
            self.proto_goal = torch.cat(self.proto_goal, axis=0)

            self.proto_goal = F.normalize(self.proto_goal, dim=1, p=2)
            
            for ix,x in enumerate(fn):
                with io.BytesIO() as bs:
                    np.savez_compressed(bs, **lst_encoded[ix])
                    bs.seek(0)
                    with x.open("wb") as f:
                        f.write(bs.read())

            self.goal_array_loaded=True

        if self.first:
            self.first=False
            self.episode_step, self.episode_reward = 0, 0
            self.time_step1 = self.train_env1.reset()
            self.meta = self.init_meta()
            self.metrics = None

            protos = self.protos.weight.data.detach().clone()
            self.goal=self.proto_goal[10][None,:]
            self.goal_key = 10
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

                if self.pos_reward:
                    self.reward =torch.as_tensor(0)
                else:
                    self.reward =torch.as_tensor(-1)
                
            self.replay_storage1.add_proto_goal(self.time_step1, self.z.cpu().numpy(), self.meta, 
                    self.goal.cpu().numpy(), self.reward, last=False)
            
        else:
            #no reward for too  long so sample goal nearby 
            if self.episode_step == self.episode_length:
                #sample new goal
                if self.episode_step!=self.step:
                    self.replay_storage1.add_proto_goal(self.time_step1,self.z.cpu().numpy(), self.meta, self.goal.cpu().numpy(), self.reward.cpu().numpy(), last=True)

                    print('keys',self.state_proto_pair.keys())
                    print('goal not reach, resample', self.step)
                    print('freq', self.goal_freq)
                    self.episode_step=0
                    self.episode_reward=0

                    #                 protos = self.protos.weight.data.detach().clone()
#                 with torch.no_grad():
#                     obs = self.time_step1.observation['pixels']
#                     obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
#                     self.z = self.encoder(obs)
#                     self.z = self.predictor(self.z)
#                     self.z = self.projector(self.z)
#                     self.z = F.normalize(self.z, dim=1, p=2)

#                     if self.pos_reward:
#                         self.reward = torch.as_tensor(0)
#                     else:
#                         self.reward =torch.as_tensor(-1)
                
                state = torch.as_tensor(self.time_step1.observation['observations'][:2])

                
                s_to_proto = torch.norm(state[None,:]- self.goal_tensor[:, :], dim=1, p=2)
                
                all_dists, _ = torch.topk(s_to_proto, self.goal_tensor.shape[0], dim=0, largest=False)
  
                if global_step<500000:
                    index=global_step//2000
                    idx = np.random.randint(max(index-10,0),min(index+20, self.proto_goal.shape[0]))
                else:
                    index=(global_step-500000)//2000
                    idx = np.random.randint(max(index-10,0),min(index+20, self.proto_goal.shape[0]))

                self.goal = self.proto_goal[_[idx]][None,:]
                self.goal_key = _[idx].item()
                print('sampled goal', self.goal_array[_[idx]])
                print('current', self.time_step1.observation['observations'])
                
            
                #ptr = self.goal_queue_ptr
                #self.goal_queue[ptr] = self.goal
                #self.goal_queue_ptr = (ptr + 1) % self.goal_queue.shape[0]
                
            if self.step==self.episode_length or self.time_step1.last():
                #make new env, switch up init_state
                print('step={}, saving last episode'.format(global_step))
                self.step=0
                if self.step!=self.episode_step:
                    self.replay_storage1.add_proto_goal(self.time_step1,self.z.cpu().numpy(), self.meta, self.goal.cpu().numpy(), self.reward.cpu().numpy(), last=True)

                if self.const_init==False and (global_step-1)%self.episode_reset_length==0:
                    initiation = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
                    initial = np.array([.15, .15])
                    init_rand = np.random.randint(4)
                    init_state = np.array([initial[0]*initiation[init_rand][0], initial[1]*initiation[init_rand][1]])
                    
                    print('reseting to', init_state)
                    self.train_env1 = dmc.make(self.task_no_goal, self.obs_type, self.frame_stack,
                                         self.action_repeat, seed=None, goal=None,
                                             init_state=(init_state[0], init_state[1]))
                    self.time_step1=self.train_env1.reset()
                else:
                    self.train_env1 = dmc.make(self.task_no_goal, self.obs_type, self.frame_stack,
                                     self.action_repeat, seed=None, goal=None, 
                                         init_state=(self.time_step1.observation['observations'][0], 
                                                     self.time_step1.observation['observations'][1]))
                
                    
                    self.time_step1 = self.train_env1.reset()
                #print('current state', self.time_step1.observation['observations'][0], self.time_step1.observation['observations'][1])
                #print('keys',self.state_proto_pair.keys())
                #print('goal not reach, resample', self.step)
                #print('freq', self.goal_freq)
                self.episode_step=0
                self.episode_reward=0

#                 protos = self.protos.weight.data.detach().clone()
#                 with torch.no_grad():
#                     obs = self.time_step1.observation['pixels']
#                     obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
#                     self.z = self.encoder(obs)
#                     self.z = self.predictor(self.z)
#                     self.z = self.projector(self.z)
#                     self.z = F.normalize(self.z, dim=1, p=2)

#                     if self.pos_reward:
#                         self.reward = torch.as_tensor(0)
#                     else:
#                         self.reward =torch.as_tensor(-1)
                
                state = torch.as_tensor(self.time_step1.observation['observations'][:2])
                
                s_to_proto = torch.norm(state[None,:]- self.goal_tensor[:, :], dim=1, p=2)
                all_dists, _ = torch.topk(s_to_proto, self.goal_tensor.shape[0], dim=0, largest=False)
  
                if global_step<500000:
                    index=global_step//2000
                    idx = np.random.randint(max(index-10,0),min(index+20, self.proto_goal.shape[0]))
                else:
                    index=(global_step-500000)//2000
                    idx = np.random.randint(max(index-10,0),min(index+20, self.proto_goal.shape[0]))

                self.goal = self.proto_goal[_[idx]][None,:]
                self.goal_key = _[idx].item()
                print('sampled goal', self.goal_array[_[idx]])
                print('current', self.time_step1.observation['observations'])
 
                protos = self.protos.weight.data.detach().clone()

                with torch.no_grad():
                    obs = self.time_step1.observation['pixels']
                    obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                    self.z = self.encoder(obs)
                    self.z = self.predictor(self.z)
                    self.z = self.projector(self.z)
                    self.z = F.normalize(self.z, dim=1, p=2)

                    if self.pos_reward:
                        self.reward =torch.as_tensor(0)
                    else:
                        self.reward =torch.as_tensor(-1)

                self.replay_storage1.add_proto_goal(self.time_step1,self.z.cpu().numpy(), self.meta, self.goal.cpu().numpy(), self.reward.cpu().numpy(),last=False)

                if self.metrics is not None:
                    # log stats
                    self._global_episode += 1
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = self.episode_step * self.action_repeat
                    with self.logger.log_and_dump_ctx(global_step*2,ty='train') as log:
                        log('fps', self.step / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', self.episode_reward)
                        log('episode_length', self.step)
                        log('episode', self._global_episode)
                        log('buffer_size', len(self.replay_storage1))
                        log('step', global_step)

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
            
            #calculate reward 
            if self.reward_nn and self.reward_scores==False:
                z_to_proto = torch.norm(self.z[:, None, :] - self.proto_goal[None, :, :], dim=2, p=2)
                all_dists, _ = torch.topk(z_to_proto, self.proto_goal.shape[0], dim=1, largest=False)

                #s_to_state = torch.norm(self.state_2[:, None, :] - self.goal_array[None, :, :], dim=2, p=2)
                #all_dists_state, _state = torch.topk(s_to_state, self.goal_array.shape[0], dim=1, largest=False)
                
                if self.pos_reward:

                    if self.goal_key==_[:,0].item():
                        self.reward=torch.as_tensor(1)
                        self.state_proto_pair[self.goal_key] = self.time_step1.observation['observations']
                        self.goal_freq[self.goal_key] += 1
                        
                    else:
                        self.reward=torch.as_tensor(0)
                    print('current', self.time_step1.observation['observations'])
                    print('goal', self.goal_array[self.goal_key])
                    print('reward', self.reward)
                else:
                    if self.goal_key==_[:,0].item():

                        self.reward=torch.as_tensor(0)
                        self.state_proto_pair[self.goal_key] = self.time_step1.observation['observations']
                        self.goal_freq[self.goal_key] += 1
                    
                    else:
                        self.reward=torch.as_tensor(-1)
                    print('current', self.time_step1.observation['observations'])
                    print('goal', self.goal_array[self.goal_key])
                    print('reward', self.reward)
                        
                    ptr = self.ten_step_queue_ptr
                    self.ten_step_reward[ptr] = self.reward
                    self.ten_step_queue_ptr = (ptr + 1) % self.ten_step_reward.shape[0]
            
            elif self.reward_nn==False and self.reward_euclid==False and self.reward_scores:
                if self.pos_reward:
                    scores = torch.matmul(self.proto_goal, self.z.T)
                    #look into why all dot products w/ proto goal are positive? but mixed w/ self.protos
                    #for x in range(self.proto_goal.shape[0]):
                    #    print(torch.dot(self.proto_goal[x,:], self.z[0,:]))
                    
                    if scores.argmax().item()==self.goal_key:
                        self.reward=torch.as_tensor(1)
                        self.state_proto_pair[self.goal_key].append(self.time_step1.observation['observations'])
                        self.goal_freq[self.goal_key] += 1
                        print('current', self.time_step1.observation['observations'])
                        print('goal', self.goal_array[self.goal_key])
                        print('reward', self.reward)
                    else:
                        self.reward=torch.as_tensor(0)


                else:
                    scores = torch.matmul(self.proto_goal, self.z.T)
                    if scores.argmax().item()==self.goal_key:
                        self.reward=torch.as_tensor(0)
                        self.state_proto_pair[self.goal_key] = self.time_step1.observation['observations']
                        self.goal_freq[self.goal_key] += 1
                    else:
                        self.reward=torch.as_tensor(-1)

                    ptr = self.ten_step_queue_ptr
                    self.ten_step_reward[ptr] = self.reward
                    self.ten_step_queue_ptr = (ptr + 1) % self.ten_step_reward.shape[0]
                    
            elif self.reward_nn==False and self.reward_euclid==False and self.reward_scores_dense:
                    scores = torch.matmul(self.proto_goal, self.z.T)
                    scores = scores/scores.max()
                    scores = scores.clamp(min=0, max=1)
                    self.reward = scores[self.goal_key]

                    if self.reward==1:
                        self.state_proto_pair[self.goal_key].append(self.time_step1.observation['observations'])
                        self.goal_freq[self.goal_key] += 1
                    
                    if self.reward >.95:
                        print('current', self.time_step1.observation['observations'])
                        print('goal', self.goal_array[self.goal_key])
                        print('reward', self.reward)
                        

            elif self.reward_euclid and self.reward_nn==False:  
                
#                 z_to_q = torch.norm(self.z[:, None, :] - self.proto_goal, dim=2, p=2)
#                 all_dists, _ = torch.topk(z_to_q, self.proto_goal.shape[0], dim=1, largest=False)
#                 idx = torch.nonzero(_==self.goal_key)
#                 dist = -(all_dists[idx[0,0],idx[0,1]]-all_dists[0,0])/(all_dists[0,self.proto_goal.shape[0]-1]-all_dists[0,0])
                
#                 tmp = 1-action1**2
#                 control_reward=max(min(tmp[0],1),0)/2
#                 control_reward+=max(min(tmp[1],1),0)/2
#                 dist_to_target=(all_dists[idx[0,0],idx[0,1]]-all_dists[0,0])/(all_dists[0,self.proto_goal.shape[0]-1]-all_dists[0,0])
                
                current=self.time_step1.observation['observations'][:2]
                goal = self.goal_array[self.goal_key]
                dist_to_target=np.linalg.norm(goal-current)
                
                if dist_to_target<.015:
                    self.reward=torch.as_tensor(1)
                else:
                    upper=.015
                    margin=.1
                    scale=torch.as_tensor(np.sqrt(2))
                    x=(dist_to_target-upper)/margin
                    self.reward=torch.exp(-.5*(x*scale)**2).long()
                    
                print('current', self.time_step1.observation['observations'])
                print('goal', self.goal_array[self.goal_key])
                print('reward', self.reward) 
                #if self.pos_reward:
                #    self.reward=dist+1
                #else:
                #    self.reward = dist

                #    ptr = self.ten_step_queue_ptr
                #    self.ten_step_reward[ptr] = self.reward
                #    self.ten_step_queue_ptr = (ptr + 1) % self.ten_step_reward.shape[0]
            else:
                self.reward=None
                print('reward function not set properly')
            
            self.episode_reward += self.reward

            if self.step!=self.episode_length and self.time_step1.last()==False:
                self.replay_storage1.add_proto_goal(self.time_step1,self.z.cpu().numpy(), self.meta, self.goal.cpu().numpy(), self.reward.cpu().numpy())

            if not self.seed_until_step(global_step):
                self.metrics = self.update(self.replay_iter1, global_step, actor1=True)
                self.logger.log_metrics(self.metrics, global_step*2, ty='train')
                
            idx = None
            
            if (self.reward_nn and self.pos_reward and self.episode_reward > 10) or (self.reward_scores and self.pos_reward and self.episode_reward > 10) or (self.reward_euclid and self.episode_reward>100) or (self.reward_nn and self.neg_reward and self.ten_step_reward.sum().item()==0) or (self.reward_scores and self.neg_reward and self.ten_step_reward.sum().item()==0) or (self.reward_scores_dense and self.episode_reward>50 and self.reward==1):
                
                self.episode_step, self.episode_reward = 0, 0
                print('reached.sampling new goal', self.step)

                                    #                 protos = self.protos.weight.data.detach().clone()
#                 with torch.no_grad():
#                     obs = self.time_step1.observation['pixels']
#                     obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
#                     self.z = self.encoder(obs)
#                     self.z = self.predictor(self.z)
#                     self.z = self.projector(self.z)
#                     self.z = F.normalize(self.z, dim=1, p=2)

#                     if self.pos_reward:
#                         self.reward = torch.as_tensor(0)
#                     else:
#                         self.reward =torch.as_tensor(-1)
                
                state = torch.as_tensor(self.time_step1.observation['observations'][:2])

                
                s_to_proto = torch.norm(state[None,:]- self.goal_tensor[:, :], dim=1, p=2)
                
                all_dists, _ = torch.topk(s_to_proto, self.goal_tensor.shape[0], dim=0, largest=False)
  
                if global_step<500000:
                    index=global_step//2000
                    idx = np.random.randint(max(index-10,0),min(index+20, self.proto_goal.shape[0]))
                else:
                    index=(global_step-500000)//2000
                    idx = np.random.randint(max(index-10,0),min(index+20, self.proto_goal.shape[0]))

                self.goal = self.proto_goal[_[idx]][None,:]
                self.goal_key = _[idx].item()
                print('sampled goal', self.goal_array[_[idx]])
                print('current', self.time_step1.observation['observations'])
                       
            if self.eval_every_step(global_step) and global_step!=0:
                
                if global_step < self.eval_after_step:
                    self.eval_heatmap_only(global_step)
                else:
                    self.eval_pairwise(global_step)
                #self.eval(global_step)
            self.episode_step += 1
            self.step +=1   

    def heatmaps(self, env, model_step, replay_dir2, goal,model_step_lb=False, gc=False,proto=False):
        if gc:
        
            heatmap = self.replay_storage1.state_visitation_gc
            print('heatmap', heatmap) 
            plt.clf()
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
            ax.set_title(model_step)

            plt.savefig(f"./{model_step}_gc_heatmap.png")
            wandb.save(f"./{model_step}_gc_heatmap.png")
            
            
            heatmap_pct = self.replay_storage1.state_visitation_gc_pct
        
            plt.clf()
            fig, ax = plt.subplots(figsize=(10,10))
            labels = np.round(heatmap_pct.T/heatmap_pct.sum()*100, 1)
            sns.heatmap(np.log(1 + heatmap_pct.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
            ax.set_title(model_step)

            plt.savefig(f"./{model_step}_gc_heatmap_pct.png")
            wandb.save(f"./{model_step}_gc_heatmap_pct.png")
            
            
            reward_matrix = self.replay_storage1.reward_matrix
            plt.clf()
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(np.log(1 + reward_matrix.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
            ax.set_title(model_step)

            plt.savefig(f"./{model_step}_gc_reward.png")
            wandb.save(f"./{model_step}_gc_reward.png")
            
        if proto:
            
            heatmap = self.replay_storage2.state_visitation_proto
        
            plt.clf()
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
            ax.set_title(model_step)

            plt.savefig(f"./{model_step}_proto_heatmap.png")
            wandb.save(f"./{model_step}_proto_heatmap.png")
            
            
            heatmap_pct = self.replay_storage2.state_visitation_proto_pct
        
            plt.clf()
            fig, ax = plt.subplots(figsize=(10,10))
            labels = np.round(heatmap_pct.T/heatmap_pct.sum()*100, 1)
            sns.heatmap(np.log(1 + heatmap_pct.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
            ax.set_title(model_step)

            plt.savefig(f"./{model_step}_proto_heatmap_pct.png")
            wandb.save(f"./{model_step}_proto_heatmap_pct.png")
        
    def eval_heatmap_only(self, global_step):

        #if global_step<=self.cut_off:
        self.heatmaps(self.eval_env, global_step, False, True,gc=True)
        #else:
        #    self.heatmaps(self.eval_env, global_step, False, True,model_step_lb=self.cut_off) 

    def eval(self, global_step):
        
        #if global_step<=self.cut_off:
        self.heatmaps(self.eval_env, global_step, False, True,gc=True)
        #else:
        #    self.heatmaps(self.eval_env, global_step, False, True,model_step_lb=self.cut_off)
        protos = self.protos.weight.data.detach().clone()
        
        for ix in range(10):
            step, episode, total_reward = 0, 0, 0
            #ininn.random.uniform((-0.29, .29),size=2)
            init = np.array([-.15, .15])
            init_state = (init[0], init[1])
            self.eval_env = dmc.make(self.task_no_goal, self.obs_type, self.frame_stack,
                    self.action_repeat, seed=None, goal=None, init_state=init_state)
            eval_until_episode = utils.Until(2)
            meta = self.init_meta()
            time_step = self.eval_env.reset()
            reached=np.array([0.,0.])
            with torch.no_grad():
                obs = time_step.observation['pixels']
                obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
                z = self.encoder(obs)
                z = self.predictor(z)
                z = self.projector(z)
                z = F.normalize(z, dim=1, p=2)

            z_to_proto = torch.norm(z[:, None, :] - protos[None, :, :], dim=2, p=2)
            print('ztop', z_to_proto.shape)
            all_dists, _ = torch.topk(z_to_proto, 50, dim=1, largest=False)
            print('all dist', all_dists.shape)
            print('_', _.shape)
            idx = np.arange(1,50,5)
            df = pd.DataFrame(columns=['x','y','r'], dtype=np.float64)
            print('ix', ix)
            for i, z in enumerate(idx):
                
                print('_', _.shape)
                iz = _[:,-z]
                goal = protos[iz, :]
                print('iz', iz)
                step, episode, total_reward = 0, 0, 0
                
                while eval_until_episode(episode):
                    time_step = self.eval_env.reset()
                    self.video_recorder.init(self.eval_env, enabled=(episode == 0))
                    
                    while step!=self.episode_length:
                        with torch.no_grad(),utils.eval_mode(self):
                            obs = torch.as_tensor(time_step.observation['pixels'].copy(), device=self.device).unsqueeze(0)
                            obs = self.encoder(obs)
                            obs = self.predictor(obs)
                            obs = self.projector(obs)
                            obs = F.normalize(obs, dim=1, p=2)
                            action = self.act(obs,
                                                goal,
                                                meta,
                                                global_step,
                                                eval_mode=True)
                        
                        #print('action', action)
                        time_step = self.eval_env.step(action)
                        #print('ts', time_step.observation['observations'])
                        self.video_recorder.record(self.eval_env)
                        if self.reward_nn and self.reward_scores==False:
                            obs_to_p = torch.norm(obs[:, None, :] - protos[None, :, :], dim=2, p=2)
                            dists, dists_idx = torch.topk(obs_to_p, 1, dim=1, largest=False)
                        
                            if torch.all(goal.eq(protos[dists_idx])):
                                reward=1
                                reached = time_step.observation['observations']
                        
                            else:
                                reward=0
                        elif self.reward_nn==False and self.reward_scores:
                            reward = None
                            print('havent written code for reward_scores yet')
                        elif self.reward_nn and self.reward_scores:
                            reward = None
                            print('two reward calculation methods passed in')
                        
                        total_reward += reward
                        step += 1
                    
                    episode += 1
                    self.video_recorder.save(f'{global_step}_{ix}_{z}th_proto.mp4')
                    print('saving')
                    print(str(self.work_dir)+'/eval_{}_{}.csv'.format(global_step, ix))
                    save(str(self.work_dir)+'/eval_{}_{}.csv'.format(global_step, ix), [[reached, total_reward, init, z]])
                
                df.loc[i, 'x'] = reached[0]
                df.loc[i, 'y'] = reached[1]
                df.loc[i, 'r'] = total_reward
                
            result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['r']/2
            print(result)
            plt.clf()
            fig, ax = plt.subplots()
            sns.heatmap(result, cmap="Blues_r").invert_yaxis()
            plt.savefig(f"./{global_step}_{init}_heatmap.png")
            wandb.save(f"./{global_step}_{init}_heatmap.png")     
            

    def eval_all_proto(self, global_step):

        #if global_step<=self.cut_off:
        self.heatmaps(self.eval_env, global_step, False, True,gc=True)
       # else:
       #     self.heatmaps(self.eval_env, global_step, False, True, model_step_lb=self.cut_off)
        protos = self.protos.weight.data.detach().clone()
        
        for ix in range(protos.shape[0]):
            step, episode, total_reward = 0, 0, 0
            init = np.random.uniform((-0.29, .29),size=2)
            init_state = (init[0], init[1])
            self.eval_env = dmc.make(self.task_no_goal, self.obs_type, self.frame_stack,
                    self.action_repeat, seed=None, goal=None, init_state=init_state)
            
            eval_until_episode = utils.Until(2)
            meta = self.init_meta()
            time_step = self.eval_env.reset()
            reached=np.array([0.,0.])
            df = pd.DataFrame(columns=['x','y','r'], dtype=np.float64)
            goal = protos[ix][None, :]
            step, episode, total_reward = 0, 0, 0
            
            while eval_until_episode(episode):
                time_step = self.eval_env.reset()
                self.video_recorder.init(self.eval_env, enabled=(episode == 0))
                
                while step!=self.episode_length:
                    with torch.no_grad(), utils.eval_mode(self):
                        obs = torch.as_tensor(time_step.observation['pixels'].copy(), device=self.device).unsqueeze(0)
                        obs = self.encoder(obs)
                        obs = self.predictor(obs)
                        obs = self.projector(obs)
                        obs = F.normalize(obs, dim=1, p=2)
                        action = self.act(obs,
                                            goal,
                                            meta,
                                            global_step,
                                            eval_mode=True)
                        
                    time_step = self.eval_env.step(action)
                    self.video_recorder.record(self.eval_env)
                    if self.reward_nn and self.reward_scores==False:

                        obs_to_p = torch.norm(obs[:, None, :] - protos[None, :, :], dim=2, p=2)
                        dists, dists_idx = torch.topk(obs_to_p, 1, dim=1, largest=False)
                    
                        if torch.all(goal.eq(protos[dists_idx])):
                            reward=1
                            reached = time_step.observation['observations']
                            print('saving')
                            print(str(self.work_dir)+'/eval_{}_{}.csv'.format(global_step, ix))
                            save(str(self.work_dir)+'/eval_{}_{}.csv'.format(global_step, ix), [[time_step.observation['observations'][:2], total_reward, init, ix]])
                        
                        else:
                            reward=0
                    elif self.reward_scores and self.reward_nn==False:
                        reward = None
                        print('havent written code for reward_scores yet')
                    elif self.reward_nn and self.reward_scores:
                        reward = None
                        print('two reward calculation methods passed in') 
                    total_reward += reward
                    step += 1
                episode += 1
                
                self.video_recorder.save(f'{global_step}_{ix}th_proto.mp4')
            df.loc[ix, 'x'] = reached[0]
            df.loc[ix, 'y'] = reached[1]
            df.loc[ix, 'r'] = total_reward
            
        result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['r']/2
        print(result)
        plt.clf()
        fig, ax = plt.subplots()
        sns.heatmap(result, cmap="Blues_r").invert_yaxis()
        plt.savefig(f"./{global_step}_{ix}_heatmap.png")
        wandb.save(f"./{global_step}_{ix}_heatmap.png")
        
        
    def eval_pairwise(self, global_step):
        save_video_idx = np.random.randint(0,self.num_protos)        
        self.heatmaps(self.eval_env, global_step, False, True,gc=True,proto=False)

        #row: init, col: goal
        df = pd.DataFrame(index=range(self.goal_array.shape[0]), columns=range(self.goal_array.shape[0]), dtype=np.float64)
        df_dist = pd.DataFrame(index=range(self.goal_array.shape[0]), columns=range(self.goal_array.shape[0]), dtype=np.float64)
        encoded = defaultdict(list)
        rand = np.random.choice(self.goal_array.shape[0], 100, replace=False) 
        for i in rand:
            print('i',i)
            step, episode, total_reward = 0, 0, 0
            encoded_i = False
            
            init = self.goal_array[i]
            init_state = (init[0], init[1])
            self.eval_env = dmc.make(self.task_no_goal, self.obs_type, self.frame_stack,
                    self.action_repeat, seed=None, goal=None, init_state=init_state)
            
            eval_until_episode = utils.Until(2)
            meta = self.init_meta()
            time_step = self.eval_env.reset()
            
            for ix in rand:
            
                goal_state = self.goal_array[ix]
                goal=self.proto_goal[ix][None,:]
                step, episode, total_reward = 0, 0, 0

                while eval_until_episode(episode):
                    
                    time_step = self.eval_env.reset()
                    
                    if ix== save_video_idx:
                        self.video_recorder.init(self.eval_env, enabled=(episode == 0))

                    while step!=500:
                        with torch.no_grad(), utils.eval_mode(self):
                            obs = torch.as_tensor(time_step.observation['pixels'].copy(), device=self.device).unsqueeze(0)
                            obs = self.encoder(obs)
                            obs = self.predictor(obs)
                            obs = self.projector(obs)
                            obs = F.normalize(obs, dim=1, p=2)
                            action = self.act(obs,
                                                goal,
                                                meta,
                                                global_step,
                                                eval_mode=True)

                        time_step = self.eval_env.step(action)
                        
                        if ix== save_video_idx:
                            self.video_recorder.record(self.eval_env)
                        
                        if self.reward_nn and self.reward_scores==False:
                            
                            z_to_proto = torch.norm(obs[:, None, :] - goal[None, :, :], dim=2, p=2)
                            all_dists, _ = torch.topk(z_to_proto, goal.shape[0], dim=1, largest=False)

                            #s_to_state = torch.norm(self.state_2[:, None, :] - self.goal_array[None, :, :], dim=2, p=2)
                            #all_dists_state, _state = torch.topk(s_to_state, self.goal_array.shape[0], dim=1, largest=False)

                            if ix==_[:,0].item():
                                total_reward+=1
                                print('reached')
                                break

                        elif self.reward_nn==False and self.reward_euclid==False and self.reward_scores:
                            scores = torch.matmul(self.proto_goal, obs.T)
                            if scores.argmax().item()==ix:
                                total_reward+=1
                                break
                                
                        elif self.reward_nn==False and self.reward_euclid==False and self.reward_scores_dense:
                            scores = torch.matmul(self.proto_goal, obs.T)
                            scores = scores/scores.max()
                            scores = scores.clamp(scores, min=0, max=1)
                            print('assuming (1,)', scores[[ix]].shape)
                            reward = scores[[ix]]
                            if reward==1:
                                total_reward+=1
                                break

                        elif self.reward_euclid and self.reward_nn==False:
                            
                            current=time_step.observation['observations'][:2]
                            goal_ = self.goal_array[self.goal_key]
                            dist_to_target=np.linalg.norm(goal_-current)

                            if dist_to_target<.015:
                                self.reward=torch.as_tensor(1)
                            else:
                                upper=.015
                                margin=.1
                                scale=torch.as_tensor(np.sqrt(2))
                                x=(dist_to_target-upper)/margin
                                self.reward=torch.exp(-.5*(x*scale)**2).long()
                                
                        step += 1
                    
                    episode += 1
                    
                    if ix== save_video_idx:
                        self.video_recorder.save(f'{global_step}_{ix}th_proto.mp4')
                        
                df.loc[i, ix] = total_reward
            
            state = torch.as_tensor(self.goal_array[i])
            s_to_proto = torch.norm(state[None,:]- self.goal_tensor[:, :], dim=1, p=2)
            all_dists, _ = torch.topk(s_to_proto, self.goal_tensor.shape[0], dim=0, largest=False)
            
            df.fillna(0,inplace=True)
            
            for z in range(self.goal_array.shape[0]):
                #starting from closest
                df_dist.loc[i,z] = df.iloc[i,_[z].item()]
        
        df_dist.fillna(0, inplace=True)
        df.to_csv(self.work_dir /'proto_pairwise_eval.csv', index=False)
        df_dist.to_csv(self.work_dir /'proto_pairwise_eval_sorted_dist.csv', index=False)
        
        plt.clf()
        fig, ax = plt.subplots()
        sns.heatmap(df_dist, cmap="Blues_r").invert_yaxis()
        plt.savefig(f"./pairwise_heatmap.png")
        wandb.save(f"./pairwise_heatmap.png")
        
        final_encoded = dict()
        
        for x in encoded.keys():
            if x == 'proto_space':
                final_encoded[x] = np.array([i.cpu().numpy() for i in encoded[x]])
            else:
                final_encoded[x] = np.array(encoded[x])
        fn = self.work_dir / 'encoded_proto.npz'
        
        with io.BytesIO() as bs:
            np.savez_compressed(bs, **final_encoded)
            bs.seek(0)
            with fn.open("wb") as f:
                f.write(bs.read())
                
    def ndim_grid(self, ndims, space):
        L = [np.linspace(-.25,.25,space) for i in range(ndims)]
        return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T
                
    def pairwise_eval_mode(self, global_step, model_name):
        save_video_idx = np.random.randint(0,self.num_protos)        
        #self.heatmaps(self.eval_env, global_step, False, True,gc=True,proto=False)
        goal_array = self.ndim_grid(2,20)
        protos = self.protos.weight.data.detach().clone()
        #row: init, col: goal
        df = pd.DataFrame(index=range(protos.shape[0]), columns=range(protos.shape[0]), dtype=np.float64)
        df_dist = pd.DataFrame(index=range(protos.shape[0]), columns=range(protos.shape[0]), dtype=np.float64)      
        encoded = defaultdict(list)
        
        for i in range(goal_array.shape[0]):
            print('i',i)
            step, episode, total_reward = 0, 0, 0
            encoded_i = False
            init=goal_array[i]
            init_state = (init[0], init[1])
            self.eval_env = dmc.make(self.task_no_goal, self.obs_type, self.frame_stack,
                    self.action_repeat, seed=None, goal=None, init_state=init_state)
            
            eval_until_episode = utils.Until(2)
            meta = self.init_meta()
            time_step = self.eval_env.reset()
            
            with torch.no_grad(), utils.eval_mode(self):
                obs = torch.as_tensor(time_step.observation['pixels'].copy(), device=self.device).unsqueeze(0)
                obs = self.encoder(obs)
                obs = self.predictor(obs)
                obs = self.projector(obs)
                obs = F.normalize(obs, dim=1, p=2)
                
            obs_to_proto = torch.norm(obs[:, None, :] - protos[None, :, :], dim=2, p=2)
            all_dists, _ = torch.topk(obs_to_proto, 1, dim=1, largest=False)
            idx = _.item()
            
            
            for ix in range(protos.shape[0]):
                print('ix',ix)
                goal = protos[ix][None, :]
                step, episode, total_reward = 0, 0, 0

                while eval_until_episode(episode):
                    time_step = self.eval_env.reset()
                    if ix== save_video_idx:
                        self.video_recorder.init(self.eval_env, enabled=(episode == 0))

                    while step<200:
                        with torch.no_grad(), utils.eval_mode(self):
                            obs = torch.as_tensor(time_step.observation['pixels'].copy(), device=self.device).unsqueeze(0)
                            obs = self.encoder(obs)
                            obs = self.predictor(obs)
                            obs = self.projector(obs)
                            obs = F.normalize(obs, dim=1, p=2)
                            action = self.act(obs,
                                                goal,
                                                meta,
                                                global_step,
                                                eval_mode=True)
                            
                        if encoded_i == False:
                            encoded['state'].append(init)
                            encoded['proto_space'].append(obs)
                            encoded_i = True

                        time_step = self.eval_env.step(action)
                        if ix== save_video_idx:
                            self.video_recorder.record(self.eval_env)
                        #if self.reward_nn and self.reward_scores==False:
#
                        obs_to_p = torch.norm(obs[:, None, :] - protos[None, :, :], dim=2, p=2)
                        dists, dists_idx = torch.topk(obs_to_p, 1, dim=1, largest=False)
                        
                        if torch.all(goal.eq(protos[dists_idx])):
                            reward=1
                            total_reward+=1
                            print('reached')
                            break
                        else:
                            reward=0
                        #elif self.reward_scores and self.reward_nn==False:
                        #    if self.protos(obs).argmax().item()==ix:
                        #        reward=1
                        #        total_reward+=1
                        #        break
                        #    else:
                        #        reward=0
                        #    
                        #    print('reward',reward)
                        #elif self.reward_nn and self.reward_scores:
                        #    reward = None
                        #    print('two reward calculation methods passed in') 
                        step += 1
                    
                    episode += 1
                    print('add episode')
                    if ix== save_video_idx:
                        self.video_recorder.save(f'{model_name}_{ix}th_proto.mp4')
                df.loc[idx, ix] = total_reward
                print('total reward', total_reward)
                
            init_to_protos = torch.norm(encoded['proto_space'][-1] - protos[None, :, :], dim=2, p=2)
            proto_dist, proto_dist_idx = torch.topk(init_to_protos, protos.shape[0], dim=1, largest=False)
            df.fillna(0,inplace=True) 
            #import IPython as ipy; ipy.embed(colors='neutral')
            for z in range(protos.shape[0]):
                #starting from closest
                df_dist.loc[idx,z] = df.iloc[idx,proto_dist_idx[:,z].item()]
        
        df_dist.fillna(0, inplace=True)
        df.to_csv(self.work_dir /'{model_name}_proto_pairwise_eval.csv', index=False)
        df_dist.to_csv(self.work_dir /'{model_name}_proto_pairwise_eval_sorted_dist.csv', index=False)
        
        
        plt.clf()
        fig, ax = plt.subplots()
        sns.heatmap(df_dist, cmap="Blues_r").invert_yaxis()
        plt.savefig(f"./{model_name}_pairwise_heatmap.png")
        wandb.save(f"./{model_name}_pairwise_heatmap.png")
        
        final_encoded = dict()
        for x in encoded.keys():
            if x == 'proto_space':
                final_encoded[x] = np.array([i.cpu().numpy() for i in encoded[x]])
            else:
                final_encoded[x] = np.array(encoded[x])
        fn = self.work_dir / '{model_name}_encoded_proto.npz'
        with io.BytesIO() as bs:
            np.savez_compressed(bs, **final_encoded)
            bs.seek(0)
            with fn.open("wb") as f:
                f.write(bs.read())
            
            

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
        #with torch.no_grad():
        #    obs = self.aug(obs)
        #    next_obs = self.aug(next_obs)
        #    if actor1:
        #        goal = self.aug(goal)
           
        if actor1==False:
            if self.reward_free:
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
            
            #obs = obs.detach()
            #next_obs = next_obs.detach()
            #goal=goal.detach()
        
            # update critic
            metrics.update(
                self.update_critic(obs.detach(), goal.detach(), action, reward, discount,
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
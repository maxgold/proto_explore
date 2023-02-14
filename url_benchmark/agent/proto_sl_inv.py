from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch import jit
import pandas as pd
import utils
import matplotlib.pyplot as plt
from agent.ddpg_sl_inv import DDPGSLInvAgent
from numpy import inf

@jit.script
def sinkhorn_knopp(Q):
    Q -= Q.max()
    Q = torch.exp(Q).T
    Q /= Q.sum()

    r = torch.ones(Q.shape[0], device=Q.device) / Q.shape[0]
    #distribution shift
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


class ProtoSLInvAgent(DDPGSLInvAgent):
    def __init__(self, pred_dim, proj_dim, queue_size, num_protos, tau,
                 encoder_target_tau, topk, update_encoder, update_gc, offline, gc_only,
                 num_iterations, update_proto_every, update_enc_proto, update_enc_gc, update_proto_opt,
                 normalize, normalize2,**kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.encoder_target_tau = encoder_target_tau
        #self.protos_target_tau = protos_target_tau
        self.topk = topk
        self.num_protos = num_protos
        self.update_encoder = update_encoder
        self.update_gc = update_gc
        self.offline = offline
        self.gc_only = gc_only
        self.num_iterations = num_iterations
        self.pred_dim=pred_dim
        self.proto_distr = torch.zeros((1000,self.num_protos), device=self.device).long()
        self.proto_distr_max = torch.zeros((1000,self.num_protos), device=self.device)
        self.proto_distr_med = torch.zeros((1000,self.num_protos), device=self.device)
        self.proto_distr_min = torch.zeros((1000,self.num_protos), device=self.device)
        self.count=torch.as_tensor(0,device=self.device)
        self.mov_avg_5 = torch.zeros((1000,), device=self.device)
        self.mov_avg_10 = torch.zeros((1000,), device=self.device)
        self.mov_avg_20 = torch.zeros((1000,), device=self.device)
        self.mov_avg_50 = torch.zeros((1000,), device=self.device)
        self.mov_avg_100 = torch.zeros((1000,), device=self.device)
        self.mov_avg_200 = torch.zeros((1000,), device=self.device)
        self.mov_avg_500 = torch.zeros((1000,), device=self.device)
        self.chg_queue = torch.zeros((1000,), device=self.device)
        self.chg_queue_ptr = 0
        self.prev_heatmap = np.zeros((10,10))
        self.current_heatmap = np.zeros((10,10))
        self.q = torch.tensor([.01, .25, .5, .75, .99], device=self.device)
        self.update_proto_every = update_proto_every
        self.goal_queue = torch.zeros((10, 2), device=self.device)
        self.goal_queue_dist = torch.zeros((10,), device=self.device)
        self.update_enc_proto = update_enc_proto
        print('update enc by proto', update_enc_proto)
        print('update enc by gc', update_enc_gc)
        self.update_enc_gc = update_enc_gc
        print('tau', tau)
        print('it', num_iterations)
        self.normalize = normalize
        self.normalize2 = normalize2
        self.update_proto_opt= update_proto_opt

        # models
        #if self.gc_only==False:
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
        self.nxt_queue = torch.zeros(queue_size, pred_dim, device=self.device)
        self.nxt_queue_ptr = 0

        # optimizers
        #self.proto_opt = torch.optim.Adam(utils.chain(
        #    self.encoder.parameters(), self.predictor.parameters(),
        #    self.projector.parameters(), self.protos.parameters()),
        #                              lr=self.lr)

        self.proto_opt = torch.optim.Adam(self.protos.parameters(), lr=self.lr)
        self.pred_opt = torch.optim.Adam(self.predictor.parameters(), lr=self.lr)
        self.proj_opt = torch.optim.Adam(self.projector.parameters(), lr=self.lr)
        self.predictor.train()
        self.projector.train()
        self.protos.train()
        self.criterion = nn.CrossEntropyLoss() 
        #elif self.load_protos:
        #    self.protos = nn.Linear(pred_dim, num_protos,
        #                                            bias=False).to(self.device)
        #    self.predictor = nn.Linear(self.obs_dim, pred_dim).to(self.device)
        #    self.projector = Projector(pred_dim, proj_dim).to(self.device)
    def init_from(self, other):
        # copy parameters over
        print('self before', self.encoder)
        utils.hard_update_params(other.encoder, self.encoder)
        print('other encoder', other.encoder)
        print('self', self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        utils.hard_update_params(other.actor2, self.actor2)
        utils.hard_update_params(other.predictor, self.predictor)
        utils.hard_update_params(other.projector, self.projector)
        utils.hard_update_params(other.protos, self.protos)
        utils.hard_update_params(other.critic, self.critic)
        utils.hard_update_params(other.critic2, self.critic2)

    def init_model_from(self, agent):
        utils.hard_update_params(agent.encoder, self.encoder)

    def init_encoder_from(self, encoder):
        utils.hard_update_params(encoder, self.encoder)

    def init_gc_from(self,critic, actor):
        utils.hard_update_params(critic, self.critic1)
        utils.hard_update_params(actor, self.actor1)
    
    def init_protos_from(self, protos):
        utils.hard_update_params(protos.protos, self.protos)
        utils.hard_update_params(protos.predictor, self.predictor)
        utils.hard_update_params(protos.projector, self.projector)
        
    def normalize_protos(self):
        C = self.protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self.protos.weight.data.copy_(C)

    def compute_intr_reward(self, obs, obs_state, step, eval=False):
        if self.normalize2:
            self.normalize_protos()
        # find a candidate for each prototype
        with torch.no_grad():
            if eval==False:
                if self.obs_type=='states':
                    z = self.encoder(obs)
                else:
                    z, _ = self.encoder(obs)
            else:
                z = obs
            z = self.predictor(z)
            if self.normalize:
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

        #if step==10000 or step%100000==0:
        #    print('set to 0')
        #    self.goal_queue = torch.zeros((10, 2), device=self.device)
        #    self.goal_queue_dist = torch.zeros((10,), device=self.device)

        #dist_arg = self.goal_queue_dist.argsort(axis=0)

        #r, _ = torch.topk(reward,10,largest=True, dim=0)

        #if eval==False:
        #    for ix in range(10):
        #        if r[ix] > self.goal_queue_dist[dist_arg[ix]]:
        #            self.goal_queue_dist[dist_arg[ix]] = r[ix]
        #            self.goal_queue[dist_arg[ix]] = obs_state[_[ix],:2]
 
        #saving dist to see distribution for intrinsic reward
        #if step%1000 and step<300000:
        #    import IPython as ipy; ipy.embed(colors='neutral')
        #    dist_np = z_to_q
        #    dist_df = pd.DataFrame(dist_np.cpu())
        #    dist_df.to_csv(self.work_dir / 'dist_{}.csv'.format(step), index=False)  
        return reward

    def update_proto(self, obs, next_obs, step):
        metrics = dict()

        # normalize prototypes
        #self.normalize_protos()

        # online network
        if self.obs_type=='states':
            s = self.encoder(obs)
        else:
            s, _ = self.encoder(obs)
        s = self.predictor(s)
        s = self.projector(s)
        if self.normalize:
            s = F.normalize(s, dim=1, p=2)
        scores_s = self.protos(s)
        #import IPython as ipy; ipy.embed(colors='neutral')
        log_p_s = F.log_softmax(scores_s / self.tau, dim=1)
        # target network
        with torch.no_grad():
            if self.obs_type=='states':
                t = self.encoder_target(next_obs)
            else:
                t, _ = self.encoder_target(next_obs)
            t = self.predictor_target(t)
            if self.normalize:
                t = F.normalize(t, dim=1, p=2)
            scores_t = self.protos(t)

            q_t = sinkhorn_knopp(scores_t / self.tau)
 
        
        loss = -(q_t * log_p_s).sum(dim=1).mean()
        #loss2 = self.criterion(p_s, q_t)

        if self.use_tb or self.use_wandb:
            metrics['repr_loss'] = loss.item()
        

        if self.update_proto_opt and step % self.update_proto_every==0:
            self.proto_opt.zero_grad(set_to_none=True)

        self.pred_opt.zero_grad(set_to_none=True)
        self.proj_opt.zero_grad(set_to_none=True)
        self.encoder_opt.zero_grad(set_to_none=True)

        loss.backward()

        self.pred_opt.step()
        self.proj_opt.step()
        self.encoder_opt.step()
        if self.update_proto_opt and step % self.update_proto_every==0:
            self.proto_opt.step()
        return metrics
    
    def update_encoder_func(self, obs, obs_state, goal, goal_state, step):

        metrics = dict()

        if self.obs_type=='states':
            obs = self.encoder(obs)
            goal = self.encoder(goal)
        else:
            obs, _ = self.encoder(obs)
            goal, _ = self.encoder(goal)

        encoder_loss = F.mse_loss(obs, obs_state) + F.mse_loss(goal, goal_state)

        if self.use_tb or self.use_wandb:

            metrics['encoder_loss'] = encoder_loss.item()

        self.encoder_opt.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_opt.step()
        
        return metrics 


    def update(self, replay_iter, step, actor1=False, test=False):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        if actor1 and step % self.update_gc==0:
            if actor1:
                obs, obs_state, action, extr_reward, discount, next_obs, goal, goal_state = utils.to_torch(
            batch, self.device)
            if self.obs_type=='states':
                goal = goal.reshape(-1, 2).float()
            extr_reward=extr_reward.float()
            goal_state = goal_state.float()
            obs_state = obs_state.float()

                
        elif actor1==False and test:
            obs, obs_state, action, extr_reward, discount, next_obs, next_obs_state, rand_obs, rand_obs_state = utils.to_torch(
                    batch, self.device)
           
            #batch moving average tracks how many states are moving in & out of each 10x10grid between every update batch. (should divide by 2)
            obs_state = obs_state.clone().detach().cpu().numpy()
            self.current_heatmap, _, _ = np.histogram2d(obs_state[:, 0], obs_state[:, 1], bins=10, range=np.array(([-.29, .29],[-.29, .29])))
            if self.prev_heatmap.sum()==0:
                self.prev_heatmap=self.current_heatmap
            else:
                total_chg = np.abs((self.current_heatmap-self.prev_heatmap)).sum()
                
                chg_ptr = self.chg_queue_ptr
                self.chg_queue[chg_ptr] = torch.tensor(total_chg,device=self.device)
                self.chg_queue_ptr = (chg_ptr+1) % self.chg_queue.shape[0]
                if step>=1000 and step%1000==0:
                    
                    indices=[5,10,20,50,100,200,500]
                    sets = [self.mov_avg_5, self.mov_avg_10, self.mov_avg_20,
                            self.mov_avg_50, self.mov_avg_100, self.mov_avg_200,
                            self.mov_avg_500]
                    for ix,x in enumerate(indices):
                        if chg_ptr-x<0:
                            lst = torch.cat([self.chg_queue[:chg_ptr], self.chg_queue[chg_ptr-x:]])
                            sets[ix][self.count]=lst.mean()
                        else:
                            sets[ix][self.count]=self.chg_queue[chg_ptr-x:chg_ptr].mean()

                if step%100000==0:
                    plt.clf()
                    fig, ax = plt.subplots(figsize=(15,5))
                    labels = ['mov_avg_5', 'mov_avg_10', 'mov_avg_20', 'mov_avg_50',
                            'mov_avg_100', 'mov_avg_200', 'mov_avg_500']
                    
                    for ix,x in enumerate(indices):
                        ax.plot(np.arange(0,sets[ix].shape[0]), sets[ix].clone().detach().cpu().numpy(), label=labels[ix])
                    ax.legend()
                    
                    plt.savefig(f"batch_moving_avg_{step}.png")
                    
        elif actor1==False:
            obs, action, reward, discount, next_obs, next_obs_state = utils.to_torch(
                    batch, self.device)
        else:
            return metrics
        
        #action = action.reshape(-1,2)
        discount = discount.reshape(-1,1)
        if obs.shape[0]!=1:
            obs = obs[None,:]
        if next_obs.shape[0]!=1:
            next_obs = next_obs[None,:]
        if actor1 and goal.shape[0]!=1:
            goal = goal[None,:]

        # augment and encode
        with torch.no_grad():
            obs = self.aug(obs)
            next_obs = self.aug(next_obs)
           
        if actor1==False:

            if self.reward_free:
                metrics.update(self.update_proto(obs, next_obs, step))
                with torch.no_grad():
                    intr_reward = self.compute_intr_reward(next_obs, next_obs_state, step)

                if self.use_tb or self.use_wandb:
                    metrics['intr_reward'] = intr_reward.mean().item()
                
                reward = intr_reward
            else:
                reward = reward
                #if self.use_tb or self.use_wandb:
                    #metrics['extr_reward'] = extr_reward.mean().item()
            
            if self.use_tb or self.use_wandb:
                metrics['batch_reward'] = reward.mean().item()

            if self.obs_type=='states':
                obs = self.encoder(obs)
                next_obs = self.encoder(next_obs)
            else:
                obs,_ = self.encoder(obs)
                next_obs, _ = self.encoder(next_obs)

            if not self.update_encoder:
                obs = obs.detach()
                next_obs = next_obs.detach()
            
            if self.update_enc_proto and self.update_encoder:
                metrics.update(self.update_encoder_func(obs, next_obs, step)) 
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
                #metrics['extr_reward'] = extr_reward.mean().item()
                metrics['batch_reward'] = reward.mean().item()

            if self.obs_type=='states':
                obs = self.encoder(obs)
                next_obs = self.encoder(next_obs)
                goal = self.encoder(goal)
            else:
                obs,_ = self.encoder(obs)
                next_obs, _ = self.encoder(next_obs)
                goal, _ = self.encoder(goal)

            if not self.update_encoder:
            
                obs = obs.detach()
                next_obs = next_obs.detach()
                goal=goal.detach()
        
            if self.update_enc_gc and self.update_encoder:
                metrics.update(self.update_encoder_func(obs, obs_state, goal, goal_state, step))

            # update actor
            metrics.update(self.update_actor(obs.detach(), goal.detach(), action, step))


        return metrics

    def get_q_value(self, obs,action):
        Q1, Q2 = self.critic2(torch.tensor(obs).cuda(), torch.tensor(action).cuda())
        Q = torch.min(Q1, Q2)
        return Q

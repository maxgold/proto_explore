from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch import jit
import pandas as pd
import utils
from agent.ddpg_encoder1 import DDPGEncoder1Agent


@jit.script
def sinkhorn_knopp(Q, num):
    Q -= Q.max()
    Q = torch.exp(Q).T
    Q /= Q.sum()

    r = torch.ones(Q.shape[0], device=Q.device) / Q.shape[0]
    c = torch.ones(Q.shape[1], device=Q.device) / Q.shape[1]
    for it in range(2):
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


class ProtoLoss1Agent(DDPGEncoder1Agent):
    def __init__(self, pred_dim, proj_dim, queue_size, num_protos, tau,
                 encoder_target_tau, topk, update_encoder, update_gc, offline, gc_only,
                 num_iterations, **kwargs):
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
        #self.load_protos = load_protos

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
        
        #elif self.load_protos:
        #    self.protos = nn.Linear(pred_dim, num_protos,
        #                                            bias=False).to(self.device)
        #    self.predictor = nn.Linear(self.obs_dim, pred_dim).to(self.device)
        #    self.projector = Projector(pred_dim, proj_dim).to(self.device)

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

    def init_gc_from(self,critic, actor):
        utils.hard_update_params(critic, self.critic1)
        utils.hard_update_params(actor, self.actor1)
    
    def init_protos_from(self, protos):
        utils.hard_update_params(protos.protos, self.protos)
        utils.hard_update_params(protos.predictor, self.predictor)
        utils.hard_update_params(protos.projector, self.projector)
        utils.hard_update_params(protos.encoder, self.encoder)
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
        #saving dist to see distribution for intrinsic reward
        #if step%1000 and step<300000:
        #    import IPython as ipy; ipy.embed(colors='neutral')
        #    dist_np = z_to_q
        #    dist_df = pd.DataFrame(dist_np.cpu())
        #    dist_df.to_csv(self.work_dir / 'dist_{}.csv'.format(step), index=False)  
        return reward

    def update_proto(self, obs, next_obs, rand_obs, step):
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
            
            v = self.encoder_target(rand_obs)
            v = self.predictor_target(v)
            v = F.normalize(v, dim=1, p=2)

            scores_t = self.protos(t)
            q_t = sinkhorn_knopp(scores_t / self.tau, self.num_iterations)
        

            
        #if step%1000==0:
        #    print(torch.argmax(q_t, dim=1).unique(return_counts=True))
        
        # loss
        if step>10000:
            loss = -(q_t * log_p_s).sum(dim=1).mean() - 1 * F.mse_loss(s, v)
            
        else:
            loss = -(q_t * log_p_s).sum(dim=1).mean()

        if self.use_tb or self.use_wandb:
            metrics['repr_loss'] = loss.item()
        
        self.proto_opt.zero_grad(set_to_none=True)
        loss.backward()
        self.proto_opt.step()

        return metrics

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
            obs, action, extr_reward, discount, next_obs, rand_obs = utils.to_torch(
                    batch, self.device)
        else:
            return metrics
        
        action = action.reshape(-1,2)
        discount = discount.reshape(-1,1)
        extr_reward = extr_reward.float()

        # augment and encode
        with torch.no_grad():
            obs = self.aug(obs)
            next_obs = self.aug(next_obs)
            rand_obs = self.aug(rand_obs)
            if actor1:
                goal = self.aug(goal)
           
        if actor1==False:

            if self.reward_free:
                metrics.update(self.update_proto(obs, next_obs, rand_obs, step))
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

            obs = self.encoder(obs)
            next_obs = self.encoder(next_obs)
            goal = self.encoder(goal)

            if not self.update_encoder:
                obs = obs.detach()
                next_obs = next_obs.detach()
                goal=goal.detach()
        
            # update critic
            metrics.update(
                self.update_critic(obs.detach(), goal.detach(), action, reward, discount,
                               next_obs.detach(), step))
            # update actor
            metrics.update(self.update_actor(obs.detach(), goal.detach(), step))

            # update critic target
            utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

    def get_q_value(self, obs,action):
        Q1, Q2 = self.critic2(torch.tensor(obs).cuda(), torch.tensor(action).cuda())
        Q = torch.min(Q1, Q2)
        return Q

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
from agent.ddpg import DDPGAgent
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


class ProtoAgent(DDPGAgent):
    def __init__(self, pred_dim, proj_dim, queue_size, num_protos, tau,
                 encoder_target_tau, topk, update_encoder, update_gc,
                 num_iterations, update_proto_every, update_enc_proto, update_enc_gc, update_proto_opt,
                 normalize, normalize2, sl, gc_inv, gym1, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.encoder_target_tau = encoder_target_tau
        #self.protos_target_tau = protos_target_tau
        self.topk = topk
        self.num_protos = num_protos
        self.update_encoder = update_encoder
        self.update_gc = update_gc
        self.num_iterations = num_iterations
        self.pred_dim=pred_dim
        self.count=torch.as_tensor(0,device=self.device)
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
        self.sl = sl
        self.gc_inv = gc_inv
        self.gym = gym1
        print('obs type', self.obs_type)
        if self.obs_type == 'pixels':
            self.pixels=True
        else:
            self.pixels=False

        if self.pixels:
            self.encoder_target = deepcopy(self.encoder)
        if self.gym:
            self.state_encoder_target = deepcopy(self.state_encoder)

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
                if self.gym:
                    if self.pixels:
                        z1 = self.encoder(obs)
                        z2 = self.state_encoder(obs_state)
                        z = torch.cat((z1,z2), dim=-1)
                    else:
                        z = self.state_encoder(obs)
                elif self.obs_type=='states' or self.sl is False:
                    z = self.encoder(obs)
                else:
                    z, _ = self.encoder(obs)
            else:
                z = obs
            if torch.isnan(z[0]).any():
                import IPython as ipy; ipy.embed(colors='neutral')
            z = self.predictor(z)
            if self.normalize:
                z = F.normalize(z, dim=1, p=2)
            scores = self.protos(z).T
            prob = F.softmax(scores, dim=1)
            candidates = pyd.Categorical(prob).sample()

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

    def update_proto(self, obs, next_obs, step, obs_state=None, next_obs_state=None):
        metrics = dict()
        # online network
        if self.gym:
            if self.pixels:
                s1 = self.encoder(obs)
                s2 = self.state_encoder(obs_state)
                s = torch.cat((s1,s2), dim=1)
            else:
                s = self.state_encoder(obs)
        elif self.obs_type=='states' or self.sl is False:
            s = self.encoder(obs)
        else:
            s, _ = self.encoder(obs)
        s = self.predictor(s)
        s = self.projector(s)
        if self.normalize:
            s = F.normalize(s, dim=1, p=2)
        scores_s = self.protos(s)
        log_p_s = F.log_softmax(scores_s / self.tau, dim=1)
        # target network
        with torch.no_grad():
            if self.gym:
                if self.pixels:
                    t1 = self.encoder_target(next_obs)
                    t2 = self.state_encoder_target(next_obs_state)
                    t = torch.cat((t1,t2), dim=1)
                else:
                    t = self.state_encoder_target(next_obs)
            elif self.obs_type=='states' or self.sl is False:
                t = self.encoder_target(next_obs)
            else:
                t, _ = self.encoder_target(next_obs)

            t = self.predictor_target(t)
            if self.normalize:
                t = F.normalize(t, dim=1, p=2)
            scores_t = self.protos(t)
            q_t = sinkhorn_knopp(scores_t / self.tau)
        
        loss = -(q_t * log_p_s).sum(dim=1).mean()
        if self.use_tb or self.use_wandb:
            metrics['repr_loss'] = loss.item()

        if self.update_proto_opt and step % self.update_proto_every==0:
            self.proto_opt.zero_grad(set_to_none=True)

        self.pred_opt.zero_grad(set_to_none=True)
        self.proj_opt.zero_grad(set_to_none=True)
        if self.pixels:
            self.encoder_opt.zero_grad(set_to_none=True)

        loss.backward()
        # # if torch.isnan():
        # #     import IPython as ipy; ipy.embed(colors='neutral')
        # print('proto', self.protos.weight.grad.norm().item())
        # print('pred', self.predictor.weight.grad.norm().item())
        # print('proj', [x.weight.grad.norm() for x in self.projector.trunk.children() if type(x) is torch.nn.Linear])
        # print('encoder', [x.weight.grad.norm() for x in self.encoder.convnet.children() if type(x) is torch.nn.Conv2d])
        self.pred_opt.step()
        self.proj_opt.step()
        if self.pixels:
            self.encoder_opt.step()
        if self.update_proto_opt and step % self.update_proto_every==0:
            self.proto_opt.step()
        return metrics
    

    def update(self, replay_iter, step, actor1=False, test=False):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        if actor1 and step % self.update_gc==0:
            if self.pixels:
                if actor1 and self.gc_inv:
                    #same for antmaze & dmenvs
                    obs, obs_state, action, extr_reward, discount, next_obs, next_obs_state, goal, goal_state = utils.to_torch(
                        batch, self.device)
                else:
                    obs, obs_state, action, extr_reward, discount, next_obs, goal, goal_state = utils.to_torch(
                batch, self.device)
            else:
                obs, action, extr_reward, discount, next_obs, goal = utils.to_torch(
                        batch, self.device)
                
                
            extr_reward=extr_reward.float()
            goal_state = goal_state.float()
            obs_state = obs_state.float()

        elif actor1==False:
            if self.pixels:
                if self.gym is False:
                    obs, action, reward, discount, next_obs, next_obs_state = utils.to_torch(
                        batch, self.device)
                else:
                    obs, obs_state, action, reward, discount, next_obs, next_obs_state = utils.to_torch(
                        batch, self.device)
            else:
                obs, action, reward, discount, next_obs = utils.to_torch(
                        batch, self.device)
        else:
            return metrics

        discount = discount.reshape(-1,1)

        # augment and encode
        if self.pixels:
            with torch.no_grad():
                obs = self.aug(obs)
                next_obs = self.aug(next_obs)
        elif self.pixels is False and self.gym:
            obs_state=None
            next_obs_state=None
            goal_state=None
           
        if actor1==False:
            if self.reward_free:
                if self.gym is False:
                    metrics.update(self.update_proto(obs, next_obs, step))
                else:
                    metrics.update(self.update_proto(obs, next_obs, step, obs_state=obs_state, next_obs_state=next_obs_state))
                
                with torch.no_grad():
                    intr_reward = self.compute_intr_reward(next_obs, next_obs_state, step)

                if self.use_tb or self.use_wandb:
                    metrics['intr_reward'] = intr_reward.mean().item()

                reward = intr_reward

            else:
                reward = reward
            
            if self.use_tb or self.use_wandb:
                metrics['batch_reward'] = reward.mean().item()

            if self.gym:
                if self.pixels:
                    obs = self.encoder(obs)
                    next_obs = self.encoder(next_obs)
                    obs_state = self.state_encoder(obs_state)
                    next_obs_state = self.state_encoder(next_obs_state)
                    obs = torch.cat((obs, obs_state), dim=-1)
                    next_obs = torch.cat((next_obs, next_obs_state), dim=-1)
                else:
                    obs = self.state_encoder(obs)
                    next_obs = self.state_encoder(next_obs)

            elif self.obs_type=='states' or self.sl is False:
                obs = self.encoder(obs)
                next_obs = self.encoder(next_obs)

            else:
                obs,_ = self.encoder(obs)
                next_obs, _ = self.encoder(next_obs)

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
            utils.soft_update_params(self.predictor, self.predictor_target,
                                 self.encoder_target_tau)
            if self.pixels:
                utils.soft_update_params(self.encoder, self.encoder_target,
                                                        self.encoder_target_tau)
            utils.soft_update_params(self.critic2, self.critic2_target,
                                 self.critic2_target_tau)
            
            if self.gym:
                #TODO: maybe we need a different tau for state encoder?
                utils.soft_update_params(self.state_encoder, self.state_encoder_target,
                                                    self.encoder_target_tau)

        elif actor1 and step % self.update_gc==0:

            reward = extr_reward
            if self.use_tb or self.use_wandb:
                metrics['batch_reward'] = reward.mean().item()

            if self.gym:
                if self.pixels:
                    obs = self.encoder(obs)
                    next_obs = self.encoder(next_obs)
                    goal = self.encoder(goal)
                    obs_state = self.state_encoder(obs_state)
                    next_obs_state = self.state_encoder(next_obs_state)
                    #dummy goal_state
                    goal_state = torch.ones_like(obs_state)
                    obs = torch.cat((obs, obs_state), dim=-1)
                    next_obs = torch.cat((next_obs, next_obs_state), dim=-1)
                    goal = torch.cat((goal, goal_state), dim=-1)
                else:
                    obs = self.state_encoder(obs)
                    next_obs = self.state_encoder(next_obs)
                    goal = self.state_encoder(goal)

            elif self.sl is False:
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

            if self.gc_inv is False:
                if self.update_enc_gc and self.update_encoder:
                    metrics.update(
                    self.update_critic(obs, action, reward, discount, next_obs.detach(), step))
                else:
                    metrics.update(
                    self.update_critic(obs.detach(), action, reward, discount, next_obs.detach(), step))

            # update actor
            if self.gc_inv and self.update_enc_gc and self.update_encoder:
                metrics.update(self.update_actor(obs, goal, action, step))
            else:
                metrics.update(self.update_actor(obs.detach(), goal.detach(), action, step))
            
            # update critic target
            if self.gc_inv is False:
                utils.soft_update_params(self.critic, self.critic_target,
                             self.critic_target_tau)

        return metrics

    def get_q_value(self, obs,action):
        Q1, Q2 = self.critic2(torch.tensor(obs).cuda(), torch.tensor(action).cuda())
        Q = torch.min(Q1, Q2)
        return Q

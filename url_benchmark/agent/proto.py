from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
from torch import jit

import utils
from agent.ddpg import DDPGAgent

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


class ProtoAgent(DDPGAgent):
    def __init__(self, pred_dim, proj_dim, queue_size, num_protos, tau,
                 encoder_target_tau, topk, update_encoder, goal, **kwargs):
        super().__init__(**kwargs)
        self.tau = tau
        self.encoder_target_tau = encoder_target_tau
        self.topk = topk
        self.num_protos = num_protos
        self.update_encoder = update_encoder
        self.goal = goal


        # models
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

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        utils.hard_update_params(other.predictor, self.predictor)
        utils.hard_update_params(other.projector, self.projector)
        utils.hard_update_params(other.protos, self.protos)
        if self.init_critic:
            utils.hard_update_params(other.critic, self.critic)

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
            # this score is P x B and measures how close 
            # each prototype is to the elements in the batch
            # each prototype is assigned a sampled vector from the batch
            # and this sampled vector is added to the queue
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

    def get_state_embeddings(self, states):
        with torch.no_grad():
            s = self.encoder(states)
            s = self.predictor(s)
            s = self.projector(s)
            s = F.normalize(s, dim=1, p=2)
        return s

    def visualize_prototypes(self):
        grid = np.meshgrid(np.arange(-.3,.3,.01),np.arange(-.3,.3,.01))
        grid = np.concatenate((grid[0][:,:,None],grid[1][:,:,None]), -1)
        grid = grid.reshape(-1, 2)
        grid = np.c_[grid, np.zeros_like(grid)]
        grid = torch.tensor(grid).cuda().float()
        grid_embeddings = self.get_state_embeddings(grid)
        protos = self.protos.weight.data.clone().detach()
        protos = F.normalize(protos, dim=1, p=2)
        dist_mat = torch.cdist(protos, grid_embeddings)
        closest_points = dist_mat.argmin(-1)
        return grid[closest_points, :2].cpu()
    
    def my_reward(self, action, next_obs, goal):
        tmp = 1 - action**2
        one = np.ones_like(tmp[:,0].cpu())
        zero = np.zeros_like(tmp[:,0].cpu())
        one = torch.as_tensor(one, device=self.device)
        zero = torch.as_tensor(zero, device=self.device)
        control_reward = torch.max(torch.min(tmp[:,0], one), zero)/2
        one_ = np.ones_like(tmp[:,1].cpu())
        zero_ = np.zeros_like(tmp[:,1].cpu())
        one_ = torch.as_tensor(one_, device=self.device)
        zero_ = torch.as_tensor(zero_, device=self.device)
        control_reward += torch.max(torch.min(tmp[:,1], one_), zero_) / 2
        pdist = torch.nn.PairwiseDistance(p=2)
        dist_to_target = pdist(goal, next_obs[:,:2])
        reward = torch.empty((dist_to_target.shape[0],1),device=self.device)
        
        upper = 0.015
        margin = 0.1
        scale = np.sqrt(-2 * np.log(0.1))
        x = (dist_to_target - upper) / margin
        r = np.exp(-0.5 * (x.cpu() * scale) ** 2)[:,None]
        r = torch.as_tensor(r, device=self.device)
        #import IPython as ipy; ipy.embed(colors='neutral')
        reward = r
        reward[dist_to_target < .015] = torch.ones_like(reward[dist_to_target < .015])
        final = torch.mul(reward, control_reward[:, None]).float()
        return final

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics
        
        if self.goal:
            batch = next(replay_iter)
            obs, action, extr_reward, discount, next_obs, goal = utils.to_torch(
                                batch, self.device)

        else:

            batch = next(replay_iter)
            obs, action, extr_reward, discount, next_obs = utils.to_torch(
            batch, self.device)
        
        obs = obs.reshape(-1, 4).float()
        next_obs = next_obs.reshape(-1, 4).float()
        goal = goal.reshape(-1, 2).float()
        action = action.reshape(-1, 2).float()
        extr_reward = extr_reward.reshape(-1, 1).float()
        discount = discount.reshape(-1, 1).float()

        #import IPython as ipy; ipy.embed(colors='neutral')
        ##add if state: 
        #goal = self.visualize_prototypes()
        #goal = goal.clone().detach()
        #goal = goal.repeat(2,1).cuda()
        ##else: (pixel)
        ##...
        #
        ##recalculate reward 
        ##if state:
        #    #use myreward from replay buffer
        #
        #reward = self.my_reward(action, next_obs, goal)
        #reward = reward.reshape(-1,1).float()
        ##else: (pixel)
        ##use similarity or MSE
        #
        # augment and encode
        with torch.no_grad():
            obs = self.aug(obs)
            next_obs = self.aug(next_obs)

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
            self.update_critic(obs.detach(), goal, action, extr_reward, discount,
                               next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), goal, action, step))

        # update critic target
        utils.soft_update_params(self.encoder, self.encoder_target,
                                 self.encoder_target_tau)
        utils.soft_update_params(self.predictor, self.predictor_target,
                                 self.encoder_target_tau)
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

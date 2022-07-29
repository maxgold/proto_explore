import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import functools

import utils
from dm_control.utils import rewards


class Actor(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim,hidden_dim):
        super().__init__()

        self.policy = nn.Sequential(nn.Linear(obs_dim+goal_dim, hidden_dim),
                                    nn.LayerNorm(hidden_dim), nn.Tanh(),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_dim))

        self.apply(utils.weight_init)

    def forward(self, obs, goal, std):
        print('obs', obs)
        print('goal', goal)
        mu = self.policy(torch.concat([obs,goal],-1))
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class BCAgent:
    def __init__(self,
                 name,
                 obs_shape,
                 action_shape,
                 goal_shape,
                 device,
                 lr,
                 hidden_dim,
                 batch_size,
                 stddev_schedule,
                 use_tb,
                 has_next_action=False,
                 expert_1=None,
                 expert_2=None,
                 expert_3=None,
                 expert_4=None):
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.stddev_schedule = stddev_schedule
        self.use_tb = use_tb
        self.expert_1=expert_1
        self.expert_2=expert_2
        self.expert_3=expert_3
        self.expert_4=expert_4
        # models
        self.actor = Actor(obs_shape[0], goal_shape[0], action_shape[0],
                           hidden_dim).to(device)

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    def act(self, obs, goal, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        goal = torch.as_tensor(goal, device=self.device).unsqueeze(0).float()
        stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.actor(obs, goal, stddev)
        if eval_mode:
            action = policy.mean
        else:
            action = policy.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_actor(self, obs, goal, action, step):
        metrics = dict()
        print('obs', obs.shape)
        print('goal', goal.shape)
        stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.actor(obs, goal, stddev)
        
      #  for ix, x in enumerate([self.expert_1, self.expert_2, self.expert_3, self.expert_4]):
      #      if ix ==0:
      #          goal = torch.tensor([.15, .15]))
      #          goal = torch.broadcast_to(goal, action.Size)
        log_prob = policy.log_prob(action).sum(-1, keepdim=True)
        actor_loss = (-log_prob).mean()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_ent'] = policy.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)

        obs, action, reward, discount, next_obs, goal = utils.to_torch(
            batch, self.device)
        action = action.reshape(-1, 2).float()
        obs = obs.reshape(-1, 4).float()
        goal = goal.reshape(-1,2).float()
        self.expert_1.act(obs, step, eval_mode=True)
        #for ix, x in enumerate([self.expert_1, self.expert_2, self.expert_3, self.expert_4]):
            #if ix ==0:
            #    goal = torch.tensor([.15, .15]))
            #    action = x.act(obs, step, eval_mode=True)
            #    print(goal)
            #    metrics.update(self.update_actor(obs, action, goal, step))
            #elif ix ==1:
             #   goal = torch.tensor([-.15, .15])
              #  action = x.act(obs, step, eval_mode=True)
               # metrics.update(self.update_actor(obs, action, goal, step))
           # elif ix ==2:
            #    goal = torch.tensor([-.15, -.15])
             #   action = x.act(obs, step, eval_mode=True)
              #  metrics.update(self.update_actor(obs, action, goal, step))
           # elif ix ==3:
            #    goal = torch.tensor([.15, -.15])
             #   action = x.act(obs, step, eval_mode=True)
              #  metrics.update(self.update_actor(obs, action, goal, step))


        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update actor
        metrics.update(self.update_actor(obs, goal, action, step))

        return metrics

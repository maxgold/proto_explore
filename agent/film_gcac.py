import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils
from dm_control.utils import rewards

class DenseResidualLayer(nn.Module):

    def __init__(self, dim):
        super(DenseResidualLayer, self).__init__()
        self.linear = nn.Linear(dim)
        
        self.apply(utils.weight_init)
        
    def forward(self, x):
        identity = x
        out = self.linear(x)
        out += identity
        return out

class FiLM(nn.Module):
    def __init__(self, goal_dim, hidden_dim):
        #target_dim = shape of matrix to be adapted (x.shape, x being output 
        #for fc layers
        
        super().__init__()
        
        #shared layer for all gamms & betas 
        self.shared_layer = nn.Sequential(nn.Linear(goal_dim, hidden_dim),
                                          nn.ReLU())
        #processors (adaptation networks) & regularization lists for each of 
        #the output params
        
        #trying residual instead of linear
        self.gamma_1 = nn.Sequential(
            DenseResidualLayer(hidden_dim),
            nn.ReLU(),
            DenseResidualLayer(hidden_dim),
            nn.ReLU(),
            DenseResidualLayer(hidden_dim)
        )
        
        self.gamma_1_regularizers = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(hidden_dim), 0, 0.001),
                                                               requires_grad=True)
        
        
        self.gamma_2 = nn.Sequential(
            DenseResidualLayer(hidden_dim),
            nn.ReLU(),
            DenseResidualLayer(hidden_dim),
            nn.ReLU(),
            DenseResidualLayer(hidden_dim)
        )
        
        self.gamma_2_regularizers = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(hidden_dim), 0, 0.001),
                                                               requires_grad=True)
        
        self.beta_1 = nn.Sequential(
            DenseResidualLayer(target_dim),
            nn.ReLU(),
            DenseResidualLayer(target_dim),
            nn.ReLU(),
            DenseResidualLayer(target_dim)
        )
        
        self.beta_1_regularizers = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(hidden_dim), 0, 0.001),
                                                               requires_grad=True)
        
        self.beta_2 = nn.Sequential(
            DenseResidualLayer(hidden_dim),
            nn.ReLU(),
            DenseResidualLayer(hidden_dim),
            nn.ReLU(),
            DenseResidualLayer(hidden_dim)
        )
        
        self.beta_2_regularizers = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(hidden_dim), 0, 0.001),
                                                               requires_grad=True)
        
        self.apply(utils.weight_init)
        
    def forward(self, goal):
        x = self.shared_layer(goal)
        
        gamma1 = self.gamma_1(x).squeeze() * self.gamma_1_regularizers + torch.ones_like(self.gamma_1_regularizers)
        beta1 = self.beta_1(x).squeeze() * self,beta_1_regularizers
        gamma2 = self.gamma_2(x).squeeze() * self.gamma_2_regularizers + torch.ones_like(self.gamma_2_regularizers)
        beta2 = self.beta_2(x).squeeze() * self,beta_2_regularizers
        
        #gammas = gammas.unsqueeze(1).unsqueeze(2).expand_as(x)
        #betas = betas.unsqueeze(1).unsqueeze(2).expand_as(x)
        return (gamma1, beta1, gamma2, beta2)
    
    def regularization_term(self):
        
        l2_term = 0
        for gamma_regularizer, beta_regularizer in zip(self.gamma1_regularizers, self.beta1_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        for gamma_regularizer, beta_regularizer in zip(self.gamma2_regularizers, self.beta2_regularizers):
            l2_term += (gamma_regularizer ** 2).sum()
            l2_term += (beta_regularizer ** 2).sum()
        return l2_term


class Actor(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.film1 = FiLM(goal_dim, hidden_dim)
        self.film2 = FiLM(goal_dim, hidden_dim)


        self.apply(utils.weight_init)

    def forward(self, obs, goal, std, film_params):
        gamma1, beta1, gamma2, beta2 = film_params
        obs_goal = torch.cat([obs, goal], dim=-1)
        x = self.ln1(self.fc1(obs_goal))
        x = self._film(x, gamma1, beta1)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self._film(x, gamma2, beta2)
        x = self.relu(x)
        x = self.fc3(x)
        mu = torch.tanh(x)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist

    def _film(self, x, gamma, beta):
        #???
        #check shape
        gamma = gamma.unsqueeze(1).unsqueeze(2).expand_as(x)
        beta = beta.unsqueeze(1).unsqueeze(2).expand_as(x)
        return gamma * x + beta


class Critic(nn.Module):
    def __init__(self, obs_dim, goal_dim, action_dim, hidden_dim):
        super().__init__()

        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim + goal_dim + action_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, goal, action):
        obs_action = torch.cat([obs, goal, action], dim=-1)
        q1 = self.q1_net(obs_action)
        q2 = self.q2_net(obs_action)

        return q1, q2


class FILM_GCACAgent:
    def __init__(self,
                 name,
                 obs_shape,
                 action_shape,
                 goal_shape,
                 device,
                 lr,
                 hidden_dim,
                 critic_target_tau,
                 stddev_schedule,
                 nstep,
                 batch_size,
                 stddev_clip,
                 use_tb,
                 has_next_action=False):
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.use_tb = use_tb
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.actor = Actor(obs_shape[0], goal_shape[0], action_shape[0],
                           hidden_dim).to(device)

        self.critic = Critic(obs_shape[0], goal_shape[0], action_shape[0],
                             hidden_dim).to(device)
        self.critic_target = Critic(obs_shape[0], goal_shape[0], action_shape[0],
                                    hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.film = FiLM(goal_shape[0], hidden_dim).to(device)
        

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.film_opt = torch.optim.Adam(self.film.parameters(), lr=lr)


        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        self.film.train(training)

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

    def update_critic(self, obs, goal, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, goal, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, goal, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, goal, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return metrics

    def update_actor(self, obs, goal, action, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        film_params = self.film(goal)
        print(film_params.shape)
        policy = self.actor(obs, goal, stddev, film_params)

        Q1, Q2 = self.critic(obs, goal, policy.sample(clip=self.stddev_clip))
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        self.film_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        self.film_opt.step()
        

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_ent'] = policy.entropy().sum(dim=-1).mean().item()
            metrics['film_params'] = film_params
        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, goal = utils.to_torch(
            batch, self.device)
        obs = obs.reshape(-1, 4).float()
        next_obs = next_obs.reshape(-1, 4).float()
        goal = goal.reshape(-1, 2).float()
        action = action.reshape(-1, 2).float()
        reward = reward.reshape(-1, 1).float()
        discount = discount.reshape(-1, 1).float()
        reward = reward.float()

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs, goal, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs, goal, action, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

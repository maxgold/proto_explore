import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils
from dm_control.utils import rewards


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim,
        goal_dim,
        action_dim,
        hidden_dim,
        num_horizon=20,
        horizon_embed_dim=8,
    ):
        super().__init__()
        self.horizon_embedding = nn.Embedding(num_horizon, horizon_embed_dim)

        self.policy = nn.Sequential(
            nn.Linear(obs_dim + goal_dim + 0 * horizon_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, action_dim, bias=False),
        )

        self.apply(utils.weight_init)

    def forward(self, obs, goal, horizon, std):
        #horizon_embed = self.horizon_embedding(horizon)
        #mu = self.policy(torch.concat([obs, goal, horizon_embed],-1))
        mu = self.policy(torch.concat([obs, goal], -1))

        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist


class GCSLAgent:
    def __init__(
        self,
        name,
        obs_shape,
        action_shape,
        goal_shape,
        device,
        lr,
        hidden_dim,
        stddev_schedule,
        nstep,
        batch_size,
        stddev_clip,
        use_tb,
        has_next_action=False,
        **kwargs
    ):
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.use_tb = use_tb
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip

        # models
        self.actor = Actor(
            obs_shape[0], goal_shape[0], action_shape[0], hidden_dim, 20, 4
        ).to(device)

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.cos_loss = nn.CosineSimilarity(dim=1, eps=1e-08)
        self.num_expl_steps = 5000

        self.train()

    def get_meta_specs(self):
        return tuple()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    def act(self, obs, goal, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0).float()
        horizon = torch.as_tensor(np.zeros(1), device=self.device).unsqueeze(0)
        goal = torch.as_tensor(goal, device=self.device).unsqueeze(0).float()
        stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.actor(obs, goal, horizon, stddev)
        if eval_mode:
            action = policy.mean
        else:
            action = policy.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.detach().cpu().numpy()[0]

    def update_actor(self, obs, goal, desired_action, horizon, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.actor(obs, goal, horizon, stddev)
        # loss = torch.square(policy.mu - desired_action).mean()
        # log_prob = policy.log_prob(desired_action).sum(-1, keepdim=True)
        # actor_loss = -log_prob.mean()
        actor_loss = torch.square(desired_action - policy.loc).mean()
        # actor_loss = -self.cos_loss(desired_action, policy.loc).mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics["actor_loss"] = actor_loss.item()
        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()
            metrics["actor_ent"] = policy.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, batch, step):
        metrics = dict()

        # batch = next(replay_iter)
        obs, action, _, _, _, goal, horizon = utils.to_torch(batch, self.device)

        metrics.update(self.update_actor(obs, goal, action, horizon, step))

        return metrics


class GCSLAgent2:
    def __init__(
        self, obs_shape, action_shape, goal_shape, device, lr, hidden_dim, num_horizon=20, **kwargs
    ):
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device

        # models
        self.actor = Actor(
            obs_shape[0], goal_shape[0], action_shape[0], hidden_dim, num_horizon, 4
        ).to(device)

        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.cos_loss = nn.CosineSimilarity(dim=1, eps=1e-08)
        self.num_expl_steps = 5000

        self.train()

    def get_meta_specs(self):
        return tuple()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)

    def act(self, obs, goal, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0).float()
        horizon = torch.as_tensor(np.zeros(1), device=self.device).unsqueeze(0)
        goal = torch.as_tensor(goal, device=self.device).unsqueeze(0).float()
        #stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.actor(obs, goal, horizon, 1)
        if eval_mode:
            action = policy.mean
        else:
            action = policy.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.detach().cpu().numpy()[0]

    def update_actor(self, obs, goal, desired_action, horizon, step):
        metrics = dict()

        #stddev = utils.schedule(self.stddev_schedule, step)
        policy = self.actor(obs, goal, horizon, 1)
        # loss = torch.square(policy.mu - desired_action).mean()
        # log_prob = policy.log_prob(desired_action).sum(-1, keepdim=True)
        # actor_loss = -log_prob.mean()
        actor_loss = torch.square(desired_action - policy.loc).mean()
        # actor_loss = -self.cos_loss(desired_action, policy.loc).mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        metrics["actor_loss"] = actor_loss.item()
        metrics["baseline"] = torch.square(desired_action).mean()
        #if self.use_tb:
        #    metrics["actor_loss"] = actor_loss.item()
        #    metrics["actor_ent"] = policy.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, loader, step):
        metrics = dict()

        batch = next(loader)
        obs, action, goal, horizon = utils.to_torch(batch, self.device)

        metrics.update(self.update_actor(obs, goal, action, horizon, step))
        metrics["episode_reward"] = metrics["actor_loss"]
        metrics["episode_length"] = metrics["baseline"]

        return metrics

from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from agent.models import Actor_proto, Critic_proto, Encoder_sl, Actor_sl, Critic_sl


class DDPGSLAgent:
    def __init__(self,
                 name,
                 reward_free,
                 obs_type,
                 obs_shape,
                 action_shape,
                 goal_shape,
                 device,
                 lr,
                 feature_dim,
                 hidden_dim,
                 critic_target_tau,
                 critic2_target_tau,
                 num_expl_steps,
                 update_every_steps,
                 stddev_schedule,
                 nstep,
                 batch_size,
                 stddev_clip,
                 init_critic,
                 use_tb,
                 use_wandb,
                 meta_dim=0,
                **kwargs):
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
        print('obs', obs_shape)
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        print('lr', lr)
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.critic2_target_tau = critic2_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.solved_meta = None
        # models
        if obs_type == 'pixels':
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder_sl(obs_shape, self.feature_dim).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
            self.goal_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim
            self.goal_dim = goal_shape[0]
        
        self.actor = Actor_sl(obs_type, self.obs_dim, self.goal_dim,self.action_dim,
                           feature_dim, hidden_dim).to(device)

        self.critic = Critic_sl(obs_type, self.obs_dim, self.goal_dim,self.action_dim,
                             feature_dim, hidden_dim).to(device)
        self.critic_target = Critic_sl(obs_type, self.obs_dim, self.goal_dim, self.action_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        
        #2nd set of actor critic networks 
        self.actor2 = Actor_proto(obs_type, self.obs_dim, self.action_dim,
                           feature_dim, hidden_dim).to(device)
        self.critic2 = Critic_proto(obs_type, self.obs_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        self.critic2_target = Critic_proto(obs_type, self.obs_dim, self.action_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())


        # optimizers

        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=lr)
        else:
            self.encoder_opt = None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.actor2_opt = torch.optim.Adam(self.actor2.parameters(), lr=lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        self.train()
        self.critic_target.train()
        self.critic2_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.actor2.train(training)
        self.critic2.train(training)

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        utils.hard_update_params(other.actor2, self.actor2)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)
            utils.hard_update_params(other.critic2.trunk, self.critic2.trunk)

    def init_encoder_from(self, encoder):
        utils.hard_update_params(encoder, self.encoder)

    def init_gc_from(self,critic, actor):
        utils.hard_update_params(critic, self.critic)
        utils.hard_update_params(actor, self.actor)
 
    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def act(self, obs, goal, meta, step, eval_mode, tile=1, general=False):
        if self.obs_type=='states':
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            goal =torch.as_tensor(goal, device=self.device).unsqueeze(0)
        else:
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
            if general is False:
                goal = np.transpose(goal, (2,0,1))
            goal = torch.as_tensor(goal.copy(), device=self.device).unsqueeze(0)
            goal = torch.tile(goal, (1,tile,1,1))
        
        h, _ = self.encoder(obs)
        g, _ = self.encoder(goal)
        inputs = [h]
        inputs2 = g
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        inpt = torch.cat([inpt, inputs2], dim=-1)
        assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self.stddev_schedule, step)

        dist = self.actor(inpt, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def act2(self, obs, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h, _ = self.encoder(obs)
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        #assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor2(inpt, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def update_critic(self, obs, goal, action, reward, discount, next_obs, step):
        metrics = dict()
        inpt = torch.cat([obs, goal], axis=-1)
        inpt_nxt = torch.cat([next_obs, goal], axis=-1)
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(inpt_nxt, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(inpt_nxt, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)
        
        Q1, Q2 = self.critic(inpt, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
        
        # optimize critic
        # if self.encoder_opt is not None:
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
#         if self.encoder_opt is not None:
        self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, goal, action,step):
        metrics = dict()
        inpt = torch.cat([obs, goal], axis=-1)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(inpt, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(inpt, action,)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()
        
        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward(retain_graph=True)
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            metrics['actor_stddev'] = stddev
        return metrics

    def update_encoder(self, obs, obs_state, goal, goal_state, step):
        metrics = dict() 
        _, obs=self.encoder(obs)
        _, goal=self.encoder(goal)
        encoder_loss = F.mse_loss(obs, obs_state) + F.mse_loss(goal, goal_state)

        if self.use_tb or self.use_wandb:

            metrics['encoder_loss'] = encoder_loss.item()
        # optimize critic
#         if self.encoder_opt is not None:
#             self.encoder_opt.zero_grad(set_to_none=True)
        self.encoder_opt.zero_grad(set_to_none=True)
        encoder_loss.backward()
        self.encoder_opt.step()
        
        return metrics 


    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_iter, step, actor1=True):
        
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        
        if actor1:
            obs, obs_state, action, extr_reward, discount, next_obs, goal, goal_state = utils.to_torch(
            batch, self.device)
            if self.obs_type=='states':
                goal = goal.reshape(-1, 2).float()
            reward=extr_reward
            goal_state = goal_state.float()
            obs_state = obs_state.float()
        else:
            return metrics

#         obs = obs.reshape(-1, 4).float()
#         next_obs = next_obs.reshape(-1, 4).float()
#         action = action.reshape(-1, 2).float()
#         reward = reward.reshape(-1, 1).float()
#         discount = discount.reshape(-1, 1).float()
#         reward = reward.float()
#         goal = goal.reshape(-1,2).float()

        # augment and encode
        obs = self.aug_and_encode(obs)
        if actor1:
            goal = self.aug_and_encode(goal)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(obs.detach(), goal.detach(), action, reward, discount, next_obs.detach(), step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), goal.detach(), action, step))
        
        metrics.update(self.update_encoder(obs, obs_state, goal, goal_state, step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)
        
        return metrics

def get_q_value(self, obs,action):
    Q1, Q2 = self.critic2(torch.tensor(obs).cuda(), torch.tensor(action).cuda())
    Q = torch.min(Q1, Q2)
    return Q

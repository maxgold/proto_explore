import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils
from dm_control.utils import rewards



class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32*9*9
        #40*40*32
        #19*19*32
        #17*17*32
        #9*9*32
        
        #number of parameters:
        #(3*3*3+1)*32 = 896
        #(3*3*32+1)*32 = 9248
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride = 2),
                                     nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class Actor(nn.Module):
    def __init__(self, obs_type, obs_dim, goal_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim
        self.trunk = nn.Sequential(nn.Linear(obs_dim+goal_dim, feature_dim), 
                                    nn.LayerNorm(feature_dim), nn.Tanh())
        policy_layers = []
        policy_layers += [
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True)
        ]
        # add additional hidden layer for pixels
        if obs_type == 'pixels':
            policy_layers += [
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
        policy_layers += [nn.Linear(hidden_dim, action_dim)]

        self.policy = nn.Sequential(*policy_layers)

        self.apply(utils.weight_init)

    def forward(self, obs, goal, std):
        obs_goal = torch.cat([obs, goal], dim=-1)
        h = self.trunk(obs_goal)
        mu = self.policy(h)
        mu = torch.tanh(mu)
        std = torch.ones_like(mu) * std
        dist = utils.TruncatedNormal(mu, std)
        return dist



class Critic(nn.Module):
    def __init__(self, obs_type, obs_dim, goal_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        self.obs_type = obs_type

        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim + goal_dim, feature_dim),
                                        nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + goal_dim + action_dim, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.Tanh())
            trunk_dim = hidden_dim

        def make_q():
            q_layers = []
            q_layers += [
                nn.Linear(trunk_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            if obs_type == 'pixels':
                q_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True),
                ]
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)
        

    def forward(self, obs, goal, action):
        inpt = torch.cat([obs, goal], dim=-1) if self.obs_type == 'pixels' else torch.cat([obs, goal, action],dim=-1)
        
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)

        return q1,q2 

    

class WGCSLAgent:
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
        self.action_dim = action_shape[0]
        #self.hidden_dim = hidden_dim
        self.lr = lr
        print('lr', lr)
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.actor_target_tau = critic2_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.init_critic = init_critic
        #self.feature_dim = feature_dim
        self.solved_meta = None
        
        # models
        if obs_type == 'pixels':
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
            self.goal_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim
            self.goal_dim = goal_shape[0]

        # models
        self.actor = Actor(obs_type, self.obs_dim, self.goal_dim,self.action_dim,
                           feature_dim, hidden_dim).to(device)
        self.actor_target = Actor(obs_type, self.obs_dim, self.goal_dim,self.action_dim,
                           feature_dim, hidden_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict()) 
        self.critic = Critic(obs_type, self.obs_dim, self.goal_dim,self.action_dim,
                             feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_type, self.obs_dim, self.goal_dim, self.action_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())        
#         if obs_type == 'pixels':
#             self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
#                                                 lr=lr)
#         else:
#             self.encoder_opt = None
            
        # optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.train()
        self.actor_target.train()
        self.critic_target.train()
    
    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        utils.hard_update_params(other.actor2, self.actor2)
        if self.init_critic:
            utils.hard_update_params(other.critic.trunk, self.critic.trunk)
            utils.hard_update_params(other.critic2.trunk, self.critic2.trunk)

    def init_model_from(self, agent):
        utils.hard_update_params(agent.encoder, self.encoder)
    
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

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, goal, meta, step, eval_mode):
        if self.obs_type=='states':
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0).float()
            goal =torch.as_tensor(goal, device=self.device).unsqueeze(0).float()
        else:
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0).int()
            goal = np.transpose(goal, (2,0,1))
            goal = torch.as_tensor(goal.copy(), device=self.device).unsqueeze(0).int()
            goal = torch.tile(goal, (1,3,1,1))

        h = self.encoder(obs)
        g = self.encoder(goal)
        inputs = [h]
        inputs2 = g
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self.stddev_schedule, step)

        dist = self.actor(inpt, inputs2, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0] 

    def update_critic_actor(self, obs, goal, action, reward, discount, next_obs, offset, step):
        metrics = dict()
        
        weights = torch.pow(discount[:,0], offset)
        weights = torch.tensor(weights[:, None])
        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor_target(next_obs, goal, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, goal, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)
            target_Q = target_Q
            # clip the q value
            #clip_return = 1 / (1 - discount)
            #target_Q = torch.clamp(target_Q, -clip_return, 0)
            

        Q1, Q2 = self.critic(obs, goal, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        policy = self.actor(obs, goal, stddev)
        with torch.no_grad():
            v1, v2 = self.critic(obs, goal, policy.sample(clip=self.stddev_clip))
            v = torch.min(v1,v2)
            #v = torch.clamp(v, -clip_return, 0)
            adv = target_Q - v
            adv = torch.clamp(torch.exp(adv), 0, 10)
        weights = weights * adv
        weights = weights.float()
        actor_loss = torch.mean(weights * torch.square(torch.subtract(policy.sample(clip=self.stddev_clip), action)))
            
        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
            metrics['actor_loss'] = actor_loss.item()

        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()
        
        # optimize critic
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        return metrics

    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

    def update(self, replay_iter, step, actor1=True):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, reward, discount, next_obs, goal, offset = utils.to_torch(
            batch, self.device)
        discount=discount.float()
        action = action.float()
        reward = reward.float()
        next_obs = next_obs.float()
        offset=offset.float()
        # augment and encode
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)
            obs = self.aug_and_encode(obs)
            goal = self.aug_and_encode(goal)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic_actor(obs.detach(), goal.detach(), action, reward, discount, next_obs.detach(), offset, step))

        # update critic target
        utils.soft_update_params(self.actor, self.actor_target,
                                 self.actor_target_tau)
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics

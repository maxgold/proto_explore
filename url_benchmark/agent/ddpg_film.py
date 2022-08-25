from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5
        h = self.convnet(obs)
        h = h.view(h.shape[0], -1)
        return h

class DenseResidualLayer(nn.Module):

    def __init__(self, dim):
        super(DenseResidualLayer, self).__init__()
        self.linear = nn.Linear(dim, dim)

        self.apply(utils.weight_init)

    def forward(self, x):
        identity = x
        out = self.linear(x)
        out += identity
        return out
    
class FiLM(nn.Module):
    def __init__(self, goal_dim, feature_dim, hidden_dim):
        #target_dim = shape of matrix to be adapted (x.shape, x being output 
        #for fc layers

        super().__init__()

        #processors (adaptation networks) & regularization lists for each of 
        #the output params
        self.shared_layer1 = nn.Sequential(nn.Linear(goal_dim, feature_dim),
            nn.ReLU())
        self.shared_layer2 = nn.Sequential(
            nn.Linear(goal_dim, hidden_dim),
            nn.ReLU())

        #trying residual instead of linear
        self.gamma_1 = nn.Sequential(
            DenseResidualLayer(feature_dim),
            nn.ReLU(),
            DenseResidualLayer(feature_dim),
            nn.ReLU(),
            DenseResidualLayer(feature_dim)
        )

        self.gamma_1_regularizers = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(feature_dim), 0, 0.001),
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
            DenseResidualLayer(feature_dim),
            nn.ReLU(),
            DenseResidualLayer(feature_dim),
            nn.ReLU(),
            DenseResidualLayer(feature_dim)
        )

        self.beta_1_regularizers = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(feature_dim), 0, 0.001),
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
        x = self.shared_layer1(goal)
        gamma1 = self.gamma_1(x).squeeze() * self.gamma_1_regularizers + torch.ones_like(self.gamma_1_regularizers)
        beta1 = self.beta_1(x).squeeze() * self.beta_1_regularizers
        x = self.shared_layer2(goal)
        gamma2 = self.gamma_2(x).squeeze() * self.gamma_2_regularizers + torch.ones_like(self.gamma_2_regularizers)
        beta2 = self.beta_2(x).squeeze() * self.beta_2_regularizers
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
    def __init__(self, obs_type, obs_dim, goal_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim
        
        self.fc1 = nn.Linear(obs_dim, feature_dim)
        self.ln1 = nn.LayerNorm(feature_dim)
        self.fc2 = nn.Linear(feature_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, action_dim)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU(inplace=True)
        self.film = FiLM(goal_dim, feature_dim, hidden_dim)
        
        self.apply(utils.weight_init)
        
    def forward(self, obs, goal, std):
        gamma1, beta1, gamma2, beta2 = self.film(goal)
        x = self.ln1(self.fc1(obs))
        x = self._film(x, gamma1, beta1)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self._film(x, gamma2, beta2)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.fc4(x)
        mu = torch.tanh(x)
        std = torch.ones_like(mu) * std

        dist = utils.TruncatedNormal(mu, std)
        return dist
    
    def _film(self, x, gamma, beta):
        #check shape
       # import IPython as ipy; ipy.embed(colors='neutral')
        gamma = gamma.expand_as(x)
        beta = beta.expand_as(x)
        return gamma * x + beta


class Actor2(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()

        feature_dim = feature_dim if obs_type == 'pixels' else hidden_dim

        self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
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

    def forward(self, obs, std):
        h = self.trunk(obs)
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
                    nn.ReLU(inplace=True)
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

        return q1, q2




class Critic2(nn.Module):
    def __init__(self, obs_type, obs_dim, action_dim, feature_dim, hidden_dim):
        super().__init__()
        self.obs_type = obs_type
        if obs_type == 'pixels':
            # for pixels actions will be added after trunk
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                       nn.LayerNorm(feature_dim), nn.Tanh())
            trunk_dim = feature_dim + action_dim
        else:
            # for states actions come in the beginning
            self.trunk = nn.Sequential(
                nn.Linear(obs_dim + action_dim, hidden_dim),
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
                    nn.ReLU(inplace=True)
                ]
            q_layers += [nn.Linear(hidden_dim, 1)]
            return nn.Sequential(*q_layers)

        self.Q1 = make_q()
        self.Q2 = make_q()

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        inpt = obs if self.obs_type == 'pixels' else torch.cat([obs, action],
                                                               dim=-1)
        h = self.trunk(inpt)
        h = torch.cat([h, action], dim=-1) if self.obs_type == 'pixels' else h

        q1 = self.Q1(h)
        q2 = self.Q2(h)
        return q1, q2



class DDPGFilmAgent:
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
        self.hidden_dim = hidden_dim
        self.lr = lr
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
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
            self.goal_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim
            self.goal_dim = goal_shape[0]
        
        self.actor = Actor(obs_type, self.obs_dim, self.goal_dim,self.action_dim,
                           feature_dim, hidden_dim).to(device)

        self.critic = Critic(obs_type, self.obs_dim, self.goal_dim,self.action_dim,
                             feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_type, self.obs_dim, self.goal_dim, self.action_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        
        #2nd set of actor critic networks 
        self.actor2 = Actor2(obs_type, self.obs_dim, self.action_dim,
                           feature_dim, hidden_dim).to(device)
        self.critic2 = Critic2(obs_type, self.obs_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        self.critic2_target = Critic2(obs_type, self.obs_dim, self.action_dim,
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

    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def act(self, obs, goal, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0).float()
        goal =torch.as_tensor(goal, device=self.device).unsqueeze(0).float()
        h = self.encoder(obs)
        g = self.encoder(goal)
        inputs = [h]
        inputs2 = g
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        #assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self.stddev_schedule, step)

        dist = self.actor(inpt, inputs2, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]

    def act2(self, obs, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        h = self.encoder(obs)
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

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, goal, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, goal, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, goal, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()
        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    def update_critic2(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor2(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic2_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic2(obs, action)
        critic2_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic2_target_q'] = target_Q.mean().item()
            metrics['critic2_q1'] = Q1.mean().item()
            metrics['critic2_q2'] = Q2.mean().item()
            metrics['critic2_loss'] = critic2_loss.item()
        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic2_opt.zero_grad(set_to_none=True)
        critic2_loss.backward()
        self.critic2_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, goal, step):
        metrics = dict()
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(obs, goal, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, goal, action,)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            metrics['actor_stddev'] = stddev
        return metrics

    def update_actor2(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor2(obs, stddev)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic2(obs, action)
        Q = torch.min(Q1, Q2)

        actor2_loss = -Q.mean()

        # optimize actor
        self.actor2_opt.zero_grad(set_to_none=True)
        actor2_loss.backward()
        self.actor2_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor2_loss'] = actor2_loss.item()
            metrics['actor2_logprob'] = log_prob.mean().item()
            metrics['actor2_ent'] = dist.entropy().sum(dim=-1).mean().item()
            metrics['actor2_stddev'] = stddev
        return metrics




    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)

#    def update(self, replay_iter, step, goal):
#        metrics = dict()
#        #import ipdb; ipdb.set_trace()
#
#        if step % self.update_every_steps != 0:
#            return metrics
#
#        batch = next(replay_iter)
#        obs, action, reward, discount, next_obs = utils.to_torch(
#            batch, self.device)
#         obs = obs.reshape(-1, 4).float()
#         next_obs = next_obs.reshape(-1, 4).float()
#         action = action.reshape(-1, 2).float()
#         reward = reward.reshape(-1, 1).float()
#         discount = discount.reshape(-1, 1).float()
#         reward = reward.float()
#         goal = goal.reshape(-1,2).float()

#         # augment and encode
#         obs = self.aug_and_encode(obs)
#         with torch.no_grad():
#             next_obs = self.aug_and_encode(next_obs)

#         if self.use_tb or self.use_wandb:
#             metrics['batch_reward'] = reward.mean().item()

#         # update critic
#         metrics.update(
#             self.update_critic(obs, goal, action, reward, discount, next_obs, step))

#         # update actor
#         metrics.update(self.update_actor(obs.detach(), goal, action, step))

#         # update critic target
#         utils.soft_update_params(self.critic, self.critic_target,
#                                  self.critic_target_tau)
#         # update critic
#         metrics.update(
#             self.update_critic2(obs, action, reward, discount, next_obs, step))

#         # update actor
#         metrics.update(self.update_actor2(obs.detach(), step))

#         # update critic target
#         utils.soft_update_params(self.critic2, self.critic2_target,
#                                  self.critic2_target_tau)
#         return metrics

def get_q_value(self, obs,action):
    Q1, Q2 = self.critic2(torch.tensor(obs).cuda(), torch.tensor(action).cuda())
    Q = torch.min(Q1, Q2)
    return Q

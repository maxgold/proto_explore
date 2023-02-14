from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from agent.models import Encoder_sl, Actor_sl, Critic_sl, Actor_proto, Critic_proto


    
class LinearInverse(nn.Module):
    # NOTE: For now the input will be [robot_rotation, box_rotation, distance_bw]
    def __init__(self, obs_dim, feature_dim, action_dim, hidden_dim):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(feature_dim*2, feature_dim), # input_dim*2: For current and goal obs
            nn.ReLU(),
            nn.Linear(feature_dim, int(hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim), int(hidden_dim/4)),
            nn.ReLU(),
            nn.Linear(int(hidden_dim/4), action_dim)
        )

    def forward(self, obs, goal):
        x = torch.cat((obs, goal), dim=-1)
        x = self.model(x)
        return x


class DDPGSLInvAgent:
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
                 stddev_schedule2,
                 stddev_clip2,
                 meta_dim=0,
                 **kwargs):
        self.reward_free = reward_free
        self.obs_type = obs_type
        self.obs_shape = obs_shape
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
        self.stddev_schedule2 = stddev_schedule2
        self.stddev_clip2 = stddev_clip2
        self.init_critic = init_critic
        self.feature_dim = feature_dim
        self.feature_dim_gc = 16
        self.solved_meta = None
        print('stddev schedule', stddev_schedule)
        print('stddev_clip', stddev_clip)
        print('stddev schedule2', stddev_schedule2)
        print('stddev_clip2', stddev_clip2)
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
        
        
        self.actor = LinearInverse(self.obs_dim, feature_dim, self.action_dim, hidden_dim).to(device)

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
        self.actor2_opt = torch.optim.Adam(self.actor2.parameters(), lr=lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        self.train()
        self.critic2_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.actor2.train(training)
        self.critic2.train(training)

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        utils.hard_update_params(other.actor2, self.actor2)
        if self.init_critic:
            utils.hard_update_params(other.critic2.trunk, self.critic2.trunk)

    def init_encoder_from(self, encoder):
        utils.hard_update_params(encoder, self.encoder)

    def init_gc_from(self,critic, actor):
        utils.hard_update_params(actor, self.actor)
 
    def get_meta_specs(self):
        return tuple()

    def init_meta(self):
        return OrderedDict()

    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta

    def act(self, obs, goal, meta, step, eval_mode, tile=1, general=False):
        if self.obs_type=='states':
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0).float()
            goal =torch.as_tensor(goal, device=self.device).unsqueeze(0).float()
        else:
            obs = torch.as_tensor(obs, device=self.device).unsqueeze(0).int()
            if general==False:
                goal = np.transpose(goal, (2,0,1))
            
            goal = torch.as_tensor(goal.copy(), device=self.device).unsqueeze(0).int()
            
            if tile > 1:
                goal = torch.tile(goal, (1,tile,1,1))
        
        if self.obs_type=='states':
            h = self.encoder(obs)
            g = self.encoder(goal)
        else:
            h, _ = self.encoder(obs)
            g, _ = self.encoder(goal)
        inputs = [h]
        inputs2 = g

        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        
        inpt = torch.cat(inputs, dim=-1)
        assert obs.shape[-1] == self.obs_shape[-1]

        action = self.actor(inpt, inputs2)
        return action.cpu().numpy()[0]

    def act2(self, obs, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        if self.obs_type=='states':
            h = self.encoder(obs)
        else:
            h, _ = self.encoder(obs)
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        #assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self.stddev_schedule2, step)
        dist = self.actor2(inpt, stddev)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]


    def update_critic2(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule2, step)
            dist = self.actor2(next_obs, stddev)
            next_action = dist.sample(clip=self.stddev_clip2)
            target_Q1, target_Q2 = self.critic2_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)
        Q1, Q2 = self.critic2(obs, action)
        critic2_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic2_loss.item()
        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic2_opt.zero_grad(set_to_none=True)
        critic2_loss.backward()
        self.critic2_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics

    def update_actor(self, obs, goal, action, step):
        metrics = dict()
        model_action = self.actor(obs, goal)
        actor_loss = F.mse_loss(action, model_action)
        
        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward(retain_graph=True)
        self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
        return metrics

    def update_actor2(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule2, step)
        dist = self.actor2(obs, stddev)
        action = dist.sample(clip=self.stddev_clip2)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic2(obs, action)
        Q = torch.min(Q1, Q2)

        actor2_loss = -Q.mean()

        # optimize actor
        self.actor2_opt.zero_grad(set_to_none=True)
        actor2_loss.backward()
        self.actor2_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor2_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            metrics['actor_stddev'] = stddev
        return metrics


    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)
   
 
def get_q_value(self, obs,action):
    Q1, Q2 = self.critic2(torch.tensor(obs).cuda(), torch.tensor(action).cuda())
    Q = torch.min(Q1, Q2)
    return Q

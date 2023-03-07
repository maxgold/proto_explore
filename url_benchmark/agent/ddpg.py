from collections import OrderedDict

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from agent.models import *


class DDPGAgent:
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
                 stddev_clip,
                 init_critic,
                 use_tb,
                 use_wandb,
                 stddev_schedule2,
                 stddev_clip2,
                 meta_dim=0,
                 sl=False,
                 encoder1=False,
                 encoder2=False,
                 encoder3=False,
                 feature_dim_gc=16,
                 inv=False,
                 use_actor_trunk=False,
                 use_critic_trunk=True,
                 init_from_proto=False,
                 init_from_ddpg=False,
                 pretrained_feature_dim=16,
                 scale=None,
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
        self.feature_dim_gc = feature_dim_gc
        self.solved_meta = None
        self.sl = sl
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        self.encoder3 = encoder3
        self.inv = inv
        self.use_actor_trunk = use_actor_trunk
        self.use_critic_trunk = use_critic_trunk
        self.init_from_proto = init_from_proto
        self.init_from_ddpg = init_from_ddpg
        self.pretrained_feature_dim = pretrained_feature_dim
        self.scale = scale
        if self.init_from_ddpg or self.init_from_proto:
            self.feature_dim = self.pretrained_feature_dim

        print('stddev schedule', stddev_schedule)
        print('stddev_clip', stddev_clip)
        print('stddev schedule2', stddev_schedule2)
        print('stddev_clip2', stddev_clip2)
        # models
        if obs_type == 'pixels':
            self.aug = utils.RandomShiftsAug(pad=4)
            if self.sl:
                self.encoder = Encoder_sl(obs_shape, self.feature_dim).to(device)
            elif self.encoder1:
                self.encoder = Encoder1(obs_shape).to(device)
            elif self.encoder2:
                self.encoder = Encoder2(obs_shape).to(device)
            elif self.encoder3:
                self.encoder = Encoder3(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim + meta_dim
            self.goal_dim = self.encoder.repr_dim + meta_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0] + meta_dim
            self.goal_dim = goal_shape[0]
        
        if self.inv:
            if self.init_from_ddpg:
                self.actor = LinearInverse(self.feature_dim, self.action_dim, hidden_dim, init_from_ddpg=self.init_from_ddpg).to(device)
            elif self.obs_type == 'states':
                self.actor = LinearInverse(6, self.action_dim, hidden_dim, obs_type=self.obs_type).to(device)
            else:
                self.actor = LinearInverse(self.feature_dim, self.action_dim, hidden_dim).to(device)
            #TODO: change init_encoder to init_model; use actor2.trunk or critic2.trunk to reduce encoder dim
            self.critic = None
            self.critic_target = None
        else:
            self.actor = Actor_gc(obs_type, self.obs_dim, self.goal_dim,self.action_dim,
                           self.feature_dim, hidden_dim).to(device)
            self.critic = Critic_gc(obs_type, self.obs_dim, self.goal_dim,self.action_dim,
                             self.feature_dim, hidden_dim).to(device)
            self.critic_target = Critic_gc(obs_type, self.obs_dim, self.goal_dim, self.action_dim,
                                        self.feature_dim, hidden_dim).to(device)
            self.critic_target.load_state_dict(self.critic.state_dict())

        #2nd set of actor critic networks 
        self.actor2 = Actor_proto(obs_type, self.obs_dim, self.action_dim,
                           self.feature_dim, hidden_dim).to(device)
        self.critic2 = Critic_proto(obs_type, self.obs_dim, self.action_dim,
                             self.feature_dim, hidden_dim).to(device)
        self.critic2_target = Critic_proto(obs_type, self.obs_dim, self.action_dim,
                                    self.feature_dim, hidden_dim).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.actor_encoder = Actor_gc(obs_type, self.obs_dim, self.goal_dim,self.action_dim,
                                           self.pretrained_feature_dim, hidden_dim).to(device)
        self.critic_encoder = Critic_gc(obs_type, self.obs_dim, self.goal_dim,self.action_dim,
                                             self.pretrained_feature_dim, hidden_dim).to(device)

        # optimizers

        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(),
                                                lr=lr)
        else:
            self.encoder_opt = None

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        if self.inv is False:
            self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.actor2_opt = torch.optim.Adam(self.actor2.parameters(), lr=lr)
        self.critic2_opt = torch.optim.Adam(self.critic2.parameters(), lr=lr)

        self.train()
        if self.inv is False:
            self.critic_target.train()
        self.critic2_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.actor2.train(training)
        self.critic2.train(training)
        
        if self.inv is False:
            self.critic.train(training)

    def init_from(self, other):
        # copy parameters over
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
        utils.hard_update_params(other.actor2, self.actor2)
        if self.init_critic:
            utils.hard_update_params(other.critic2.trunk, self.critic2.trunk)

    def init_encoder_from(self, encoder):
        utils.hard_update_params(encoder, self.encoder)

    def init_encoder_trunk_from(self, encoder, critic2, actor2):
        utils.hard_update_params(encoder, self.encoder)
        utils.hard_update_params(actor2.trunk, self.actor2.trunk)
        utils.hard_update_params(critic2.trunk, self.critic2.trunk)

    def init_encoder_trunk_gc_from(self, encoder, critic, actor):
        utils.hard_update_params(encoder, self.encoder)
        utils.hard_update_params(actor.trunk, self.actor_encoder.trunk)
        utils.hard_update_params(critic.trunk, self.critic_encoder.trunk)
     
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
            inputs = [h]
            inputs2 = g
        else:
            if self.sl:
                h, _ = self.encoder(obs)
                g, _ = self.encoder(goal)
                inputs = [h]
                inputs2 = g

            elif self.init_from_proto:
                #TODO: use actor2.trunk or critic2.trunk to reduce encoder dim
                h = self.encoder(obs)
                g = self.encoder(goal)
                if self.use_actor_trunk:
                    h = self.actor2.trunk(h)
                    g = self.actor2.trunk(g)
                else:
                    h = self.critic2.trunk(h)
                    g = self.critic2.trunk(g)

                inputs = [h]
                inputs2 = g

            elif self.init_from_ddpg:
                h = self.encoder(obs)
                g = self.encoder(goal)
                if self.use_actor_trunk:
                    h = torch.cat([h, g], dim=-1)
                    h = self.actor_encoder.trunk(h)
                else:
                    h = torch.cat([h, g], dim=-1)
                    h = self.critic_encoder.trunk(h)
                
                inputs = [h]
                inputs2 = None

        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        
        inpt = torch.cat(inputs, dim=-1)
        assert obs.shape[-1] == self.obs_shape[-1]

        action = self.actor(inpt, inputs2)
        return action.cpu().numpy()[0]

    def act2(self, obs, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        if self.obs_type=='states' or self.sl is False:
            h = self.encoder(obs)
        elif self.sl:
            h, _ = self.encoder(obs)
            
        inputs = [h]
        for value in meta.values():
            value = torch.as_tensor(value, device=self.device).unsqueeze(0)
            inputs.append(value)
        inpt = torch.cat(inputs, dim=-1)
        #assert obs.shape[-1] == self.obs_shape[-1]
        stddev = utils.schedule(self.stddev_schedule2, step)
        dist = self.actor2(inpt, stddev, scale=self.scale)
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
            stddev = utils.schedule(self.stddev_schedule2, step)
            dist = self.actor2(next_obs, stddev, scale=self.scale)
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

        if self.inv:
            if self.init_from_proto:
                #TODO: use actor2.trunk or critic2.trunk to reduce encoder dim
                if self.use_actor_trunk:
                    obs = self.actor2.trunk(obs)
                    goal = self.actor2.trunk(goal)
                else:
                    obs = self.critic2.trunk(obs)
                    goal = self.critic2.trunk(goal)

            elif self.init_from_ddpg:

                if self.use_actor_trunk:
                    obs = torch.cat([obs, goal], dim=-1)
                    obs = self.actor_encoder.trunk(obs)
                else:
                    obs = torch.cat([obs, goal], dim=-1)
                    obs = self.critic_encoder.trunk(obs)
                
                goal = None

            model_action = self.actor(obs, goal)
            actor_loss = F.mse_loss(action, model_action)
        else:
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(obs, goal, stddev)
            action = dist.sample(clip=self.stddev_clip)
            log_prob = dist.log_prob(action).sum(-1, keepdim=True)
            Q1, Q2 = self.critic(obs, goal, action)
            Q = torch.min(Q1, Q2)
            actor_loss = -Q.mean()
        
        # optimize actor
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward(retain_graph=True)
        self.actor_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            #TODO: add other metrics; figure out how to add it so that we can avoid error
        return metrics

    def update_actor2(self, obs, step):
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule2, step)
        dist = self.actor2(obs, stddev, scale=self.scale)
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

    def update(self, replay_iter, step, actor1=True):
        metrics = dict()
        #import ipdb; ipdb.set_trace()

        if step % self.update_every_steps != 0:
            return metrics

        batch = next(replay_iter)
        #TODO: change this part to be compatible with the new replay buffer
        obs, obs_state, action, reward, discount, next_obs, next_obs_state, goal, goal_state = utils.to_torch(
            batch, self.device)
        

        if self.obs_type == 'states':
            obs = obs_state
            next_obs = next_obs_state
            goal = goal_state
 
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)
            goal = self.aug_and_encode(goal)

        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()

        # update critic
        if self.inv is False:
            metrics.update(
                self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs, goal, action, step))
        

        # update critic target
        if self.inv is False:
            utils.soft_update_params(self.critic, self.critic_target,
                             self.critic_target_tau)

        return metrics
   
 
def get_q_value(self, obs,action):
    Q1, Q2 = self.critic2(torch.tensor(obs).cuda(), torch.tensor(action).cuda())
    Q = torch.min(Q1, Q2)
    return Q

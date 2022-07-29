import datetime
import io
import random
import traceback
import copy
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from dm_control.utils import rewards


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def relable_episode(env, episode):
    rewards = []
    reward_spec = env.reward_spec()
    states = episode["physics"]
    for i in range(states.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(states[i])
        reward = env.task.get_reward(env.physics)
        reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
        rewards.append(reward)
    episode["reward"] = np.array(rewards, dtype=reward_spec.dtype)
    return episode


def my_reward(action, next_obs, goal):
    # this is optimized for speed ...
    tmp = 1 - action**2
    control_reward = max(min(tmp[0], 1), 0) / 2
    control_reward += max(min(tmp[1], 1), 0) / 2
    dist_to_target = np.linalg.norm(goal - next_obs[:2])
    if dist_to_target < 0.015:
        r = 10
    else:
        #upper = 0.015
        #margin = 0.1
        #scale = np.sqrt(-2 * np.log(0.1))
        #x = (dist_to_target - upper) / margin
        #r = np.exp(-0.5 * (x * scale) ** 2)
        r=-1
    return float(r * control_reward)

class ReplayBufferStorage:
    def __init__(self, data_specs, meta_specs, replay_dir):
        self._data_specs = data_specs
        self._meta_specs = meta_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step, meta, q, task):
        for key, value in meta.items():
            self._current_episode[key].append(value)
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        for i in range(length(q)):
            self._current_episode['q_value'].append(q[i])
        for i in range(length(task)):
            self._current_episode['task'].append(task)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            for spec in self._meta_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            
            value = self._current_episode['q_value']
            episode['q_value'] = np.array(value, np.float64)
            
            value = self._current_episode['task']
            episode['task'] = np.array(value, np.float64)
            
            self._current_episode = defaultdict(list)
            self._store_episode(episode)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(
        self,
        storage,
        max_size,
        num_workers,
        nstep,
        discount,
        fetch_every,
        save_snapshot,
    ):
        self._storage = storage
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._storage._replay_dir.glob("*.npz"), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        meta = []
        for spec in self._storage._meta_specs:
            meta.append(episode[spec.name][idx - 1])
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount
        return (obs, action, reward, discount, next_obs, *meta)

    def __iter__(self):
        while True:
            yield self._sample()


#GOAL_ARRAY = np.array([[-0.15, 0.15], [-0.15, -0.15], [0.15, -0.15], [0.15, 0.15]])

class OfflineReplayBuffer(IterableDataset):
    def __init__(self,
        env,
        replay_dir,
        max_size,
        num_workers,
        discount,
        offset=100,
        offset_schedule=None,
        random_goal=False,
        goal=False,
        vae=False,
        #gamma,
        #agent,
        #method,
        #baw_delta,
        #baw_max,
        #num_replay_goals
    ):

        self._env = env
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._discount = discount
        self._loaded = False
        self.offset = offset
        self.offset_schedule = offset_schedule
        self.goal = goal
        self.vae = vae
        self.goal_array = []
        self._goal_array = False
        self.obs = []
        #self.gamma = gamma
        #self.method = method
        #self.baw_delta = baw_delta
        #self.baw_max = baw_max
        #self.num_replay_goals = num_replay_goals
        
        
        #future_p = 1 - (1. / (1 + self.num_replay_goals))
        
    def _load(self, relable=True):
        #space: e.g. .2 apart for uniform observation from -1 to 1
        print("Labeling data...")
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"))
        # for eps_fn in tqdm.tqdm(eps_fns):
        obs_tmp = []
        for eps_fn in eps_fns:
            if self._size > self._max_size:
                break
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            episode = load_episode(eps_fn)
            if relable:
                episode = self._relable_reward(episode)
            self._episode_fns.append(eps_fn)
            self._episodes[eps_fn] = episode
            self._size += episode_len(episode)


    
    def _get_goal_array(self, eval_mode=False, space=6):
        #assuming max & min are 1, -1, but position vector can be 2d or more dim.
        #fix obs_dim. figure out how to index position & orientation 
    
        if eval_mode==False:
            #obs_dim = self.env.observation_spec()['position'].shape[0]
            obs_dim = 2
            self.goal_array = ndim_grid(obs_dim, space)
            #self.goal_array = np.random.uniform(low=-1, high=1, size=(4,2))
        else:
            if not self._loaded:
                self._load()
                self._loaded = True

            #obs_dim = env.observation_spec()['position'].shape[0]
            obs_dim = 2
            goal_array = ndim_grid(obs_dim, space)
            return goal_array
        

    def _sample_episode(self):
        if not self._loaded:
            self._load()
            self._loaded = True
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _relable_reward(self, episode):
        return relable_episode(self._env, episode)

    def _sample(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode)) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx]
        reward = episode["reward"][idx]
        discount = episode["discount"][idx] * self._discount
        reward = my_reward(action, next_obs, np.array((0.15, 0.15)))
        #        control_reward = rewards.tolerance(
        #            action, margin=1, value_at_margin=0, sigmoid="quadratic"
        #        ).mean()
        #        small_control = (control_reward + 4) / 5
        #        near_target = rewards.tolerance(
        #            np.linalg.norm(np.array((.15,.15)) - next_obs[:2]),
        #            bounds=(0, .015),
        #            margin=.015,
        #        )
        #        reward = near_target * small_control
        return (obs, action, reward, discount, next_obs)
    
    def _sample_sequence(self, offset=10):
        
        #len of obs should be 10 
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - offset) + 1
        obs = episode["observation"][idx - 1:idx + 9]
        action = episode["action"][idx-1:idx+9]
        next_obs = episode["observation"][idx:idx+10]
        reward = episode["action"][idx-1:idx+9]
        q_value = episode["q"][idx-1:idx+9]
        split = episode['task'][0].split('_')
       
        
        if split[-2] == 'top':
            #check these coordinates
            
            if split[-1] == 'left':
                x_coor = -.25 * np.random.sample((len(obs)))
                y_coor = .25 * np.random.sample((len(obs)))
                
            else:
                x_coor = .25 * np.random.sample((len(obs)))
                y_coor = .25 * np.random.sample((len(obs)))
        else:
            
            if split[-1] == 'left':
                x_coor = -.25 * np.random.sample((len(obs)))
                y_coor = -.25 * np.random.sample((len(obs)))
                
            else:
                x_coor = .25 * np.random.sample((len(obs)))
                y_coor = -.25 * np.random.sample((len(obs)))
        
        goal = np.array(zip(x_coor, y_coor))
        
        return (obs, action, reward, discount, next_obs, goal, q)
    

    def _sample_goal(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition

        idx = np.random.randint(0, episode_len(episode) - self.offset) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx]
        goal = episode["observation"][idx + self.offset][:2]
        rewards = []

        #sampling 5 goals from uniform goal matrix
        
        if not self._goal_array:
            self._get_goal_array()
            self._goal_array = True
        
        goal_array = random.sample(np.ndarray.tolist(self.goal_array),5)
        #goal_array = np.array([[-0.15, 0.15], [-0.15, -0.15], [0.15, -0.15], [0.15, 0.15]])
        for goal in goal_array:
            rewards.append(my_reward(action, next_obs, goal))

        discount = np.ones_like(episode["discount"][idx])
        obs = np.tile(obs, (5, 1))
        action = np.tile(action, (5, 1))
        discount = np.tile(discount, (5, 1))
        next_obs = np.tile(next_obs, (5, 1))
        reward = np.array(rewards)
        goal_array = np.array(goal_array)
        return (obs, action, reward, discount, next_obs, goal_array)


    def _sample_future(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self.offset) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self.offset]
        reward = episode["reward"][idx]
        discount = episode["discount"][idx] * self._discount
        return (obs, action, reward, discount, next_obs)
    
#     def _sample_supervised_transitions(self):
#         batch_size = 10
#         episode = self._sample_episode()
#         t_samples = np.random.randint(len(episode), size=batch_size)
#         for i in range(len(t_samples)):
#             #change data type of transitions
#             self.transitions += episode[t_samples[i]]
# #             transitions = {key: episode_batch[key][episode_idxs, t_samples].copy()
# #                             for key in episode_batch.keys()}
#             #if in range
#             self.transitions['goal'] = episode[t_samples[i]+100]
#         #what is her_indexes used for?
#         her_indexes = (np.random.uniform() < future_p)
#         #returns true or false. future_p is prob. of picking future goal?
#         offset = np.random.uniform() * (len(episode)-t_sample)
#         offset = future_offset.astype(int)
#         future_t = t_samples + 1 + offset
#         original_g = 
#         if her_indexes:
#             future_achieved_goal = episode[future_t]
#         else: 
            
#         method_lst = method.split('_')
        
#         if 'gamma' in method_lst:
#             weights = pow(gamma, offset)
#         else:
#             weights = np.ones(batch_size)
#         if 'adv' in method_lst:
#             ##fix. what is value?
#             Q1, Q2 = agent.critic(self.transitions['obs'], self.transitions['goal'], policy.sample(clip=agent.stddev_clip))
#             value = torch.min(Q1, Q2).reshape(-1)
#             Q1, Q2 = self.critic(self.transitions['next_obs'], self.transitions['goal'], policy.sample(clip=self.stddev_clip))
#             next_value = torch.min(Q1, Q2).reshape(-1)
#             adv = self.my_reward(transitions['achieved_goal_next'], transitions['goal']) + discount * next_value - value
            
#             if 'baw' in method_lis:
#                 advque.update(adv)
#                 global global_threshold
#                 global_threshold = min(global_threshold + baw_delta, baw_max)
#                 threshold = advque.get(global_threshold)'
                
#             if 'exp' in method_lis:  # exp weights
#                 if 'clip10' in method_lis:
#                     weights *= np.clip(np.exp(adv), 0, 10)
#                 elif 'clip5' in method_lis:
#                     weights *= np.clip(np.exp(adv), 0, 5)
#                 elif 'clip1' in method_lis:
#                     weights *= np.clip(np.exp(adv), 0, 1)
#                 else:
#                     weights *= np.exp(adv) 
                    
                    
#             if 'baw' in method_lis:
#                 positive = adv.copy()
#                 positive[adv >= threshold] = 1
#                 positive[adv < threshold] = 0.05
#                 weights *= positive
                
#         loss = train_policy(o=transitions['o'], g=transitions['g'], u=transitions['u'], weights=weights) 
        
#         keep_origin_rate = 0.2
#         origin_index = (np.random.uniform(size=batch_size) < keep_origin_rate)
#         transitions['g'][origin_index] = original_g[origin_index]
#         transitions['r'] = _get_reward(transitions['ag_2'], transitions['g']) 
#         ## make next_ob and ag_2 
        
#         return _sample_supervised_transitions, _sample_her_transitions
                
                
                
                
    def __iter__(self):
        while True:
            if self.goal:
                yield self._sample_goal()
            elif self.vae:
                yield self._sample_future()
            else:
                yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(
    env,
    replay_dir,
    max_size,
    batch_size,
    num_workers,
    discount,
    offset=100,
    goal=False,
    vae=False,
):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = OfflineReplayBuffer(
        env,
        replay_dir,
        max_size_per_worker,
        num_workers,
        discount,
        offset,
        goal=goal,
        vae=vae,
    )
    iterable._load(relable=False)

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader


def make_replay_loader_online(
    storage, max_size, batch_size, num_workers, save_snapshot, nstep, discount
):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(
        storage,
        max_size_per_worker,
        num_workers,
        nstep,
        discount,
        fetch_every=1000,
        save_snapshot=save_snapshot,
    )

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader


def ndim_grid(ndims, space):
    L = [np.linspace(-.25,.25,space) for i in range(ndims)]
    return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T

import datetime
import io
import random
import traceback
import copy
from collections import defaultdict
import tqdm

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


def relabel_episode(env, episode):
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
    def __init__(self, data_specs,  replay_dir):
        self._data_specs = data_specs
#         self._meta_specs = meta_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions

    def add(self, time_step, q, task, model_time_step):
#         for key, value in meta.items():
#             self._current_episode[key].append(value)
        
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        
        for i in range(len(q)):
            self._current_episode['q_value'].append(q[i])
        
        
        for i in range(len(task)):
            tmp = task.split('/')
            split = tmp[-2].split('_')
            
            if split[-1] == 'right':
                if split[-2] == 'top':
                    self._current_episode['task'].append(np.array(1.))
                    #print('top right')
                else:
                    self._current_episode['task'].append(np.array(4.))
                    #print('top left')
            elif split[-1] == 'left':
                if split[-2]=='top':
                    self._current_episode['task'].append(np.array(2.))
                    #print('top left')
                else:
                    self._current_episode['task'].append(np.array(3.))
                    #print('bottom left')

        #print(self._current_episode['task'])
        if time_step.last():    
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
#             for spec in self._meta_specs:
#                 value = self._current_episode[spec.name]
#                 episode[spec.name] = np.array(value, spec.dtype)
            
            value = self._current_episode['q_value']
            episode['q_value'] = torch.tensor(value)
            
            value = self._current_episode['task']
            episode['task'] = np.array(value, np.float64)
            
    
            self._current_episode = defaultdict(list)
            self._store_episode(episode, model_time_step)

    def _preload(self):
        self._num_episodes = 0
        self._num_transitions = 0
        for fn in self._replay_dir.glob("*.npz"):
            _, _, eps_len = fn.stem.split("_")
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode, time_step):
        print('storing now')
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        eps_fn = f"{ts}_{time_step}_{eps_idx}_{eps_len}.npz"
        print('eps_fn')
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
        distill=False,
        expert_dict=None,
        goal_dict=None,
        relabel = False
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
        self.distill = distill
        self.relabel = relabel
        
    def _load(self):
        relabel=self.relabel
        #space: e.g. .2 apart for uniform observation from -1 to 1
        print("Labeling data...")
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        
        print(self._replay_dir)
        eps_fns = sorted(self._replay_dir.glob("*.npz"))
       # print(self._replay_dir)
       # print(eps_fns)
        # for eps_fn in tqdm.tqdm(eps_fns):
        #random.shuffle(eps_fns)
        #for eps_fn in eps_fns:
        for eps_fn in tqdm.tqdm(eps_fns, disable=worker_id!=0):
            if self._size > self._max_size:
                break
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split("_")[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            episode = load_episode(eps_fn)
            if relabel:
                episode = self._relabel_reward(episode)
            self._episode_fns.append(eps_fn)
            self._episodes[eps_fn] = episode
            self._size += episode_len(episode)
            #print('size', self._size)
            #print('len', len(self._episodes))
        #import IPython as ipy; ipy.embed(colors='neutral')
    
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

    def _relabel_reward(self, episode):
        return relabel_episode(self._env, episode)

    def _sample(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode)) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx]
        reward = episode["reward"][idx]
        #print('reward', reward)
        discount = episode["discount"][idx] * self._discount
        #reward = my_reward(action, next_obs, np.array((0.15, 0.15)))
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
    
    def _sample_distill(self):
        step=1
        #eval mode step doesn't matter
        obs, action, reward, discount, next_obs = self._sample()
        key = np.random.choice(range(len(self.expert_dict.keys())))
        action = self.expert_dict[key].act(obs, step, eval_mode=True)
        goal = np.array(self.goal_dict[key])
        return (obs, action, reward, discount, next_obs, goal)

    def _sample_sequence(self, offset=10):
        #len of obs should be 10 
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - offset) + 1
        obs = episode["observation"][idx - 1:idx + 9]
        action = episode["action"][idx-1:idx+9]
        next_obs = episode["observation"][idx:idx+10]
        reward = episode["action"][idx-1:idx+9]
        q_value = episode["q_value"][idx-1:idx+9]
        discount = episode["discount"][idx-1:idx+9]
        task = episode['task'][idx-1:idx+9]
        x_coor = [] 
        y_coor = []
        goal = []
        
        if task[0] == 2.:
            #check these coordinates
            x_coor.append(-.25 * np.random.sample((len(obs))))
            y_coor.append(.25 * np.random.sample((len(obs))))
            #x_coor=np.tile(-.15, (len(task), 1))
            #y_coor=np.tile(.15, (len(task), 1))
            
        elif task[0] == 1.:
            x_coor.append(.25 * np.random.sample((len(obs))))
            y_coor.append(.25 * np.random.sample((len(obs))))
            #x_coor=np.tile(.15, (len(task), 1))
            #y_coor=np.tile(.15, (len(task), 1))
        elif task[0] == 3.:
            x_coor.append(-.25 * np.random.sample((len(obs))))
            y_coor.append(-.25 * np.random.sample((len(obs))))
            #x_coor=np.tile(-.15, (len(task), 1))
            #y_coor=np.tile(-.15, (len(task), 1))
        elif task[0] == 4.:
            x_coor.append(.25 * np.random.sample((len(obs))))
            y_coor.append(-.25 * np.random.sample((len(obs))))
            #x_coor=np.tile(.15, (len(task), 1))
            #y_coor=np.tile(-.15, (len(task), 1))
        else:
            #import IPython as ipy; ipy.embed(colors="neutral")
            print('cant determine task')
            print('task', i)

        goal.append(np.array([x_coor, y_coor]))
        goal = np.array(goal)
        #print('obs', obs.shape)
        #print('action',action.shape)
        #print('reward', reward.shape)
        #print('discount',discount.shape)
        #print('next_obs',next_obs.shape)
        #print('goal',goal.shape)
        #print('q_value',q_value.shape)
        return (obs, action, reward, discount, next_obs, goal, q_value)
    

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
        
        #goal_array = random.sample(np.ndarray.tolist(self.goal_array),5)
        goal_array = np.array([[-0.15, 0.15], [-0.15, -0.15], [0.15, -0.15], [0.15, 0.15]])
        for goal in goal_array:
            rewards.append(my_reward(action, next_obs, goal))

        #cand_goals = np.random.uniform(-.2,.2, size=(50,2))
        #cand_goals = cand_goals[np.abs(cand_goals[:,0])>.05]
        #cand_goals = cand_goals[np.abs(cand_goals[:,1])>.05]
        #cand_goals = cand_goals[:4]
        #for goal in cand_goals:
        #    rewards.append(my_reward(action, next_obs, goal))
        discount = np.ones_like(episode["discount"][idx])
        obs = np.tile(obs, (len(goal_array), 1))
        action = np.tile(action, (len(goal_array), 1))
        discount = np.tile(discount, (len(goal_array), 1))
        next_obs = np.tile(next_obs, (len(goal_array), 1))
        reward = np.array(rewards)
        goal_array = np.array(goal_array)
        return (obs, action, reward, discount, next_obs, goal_array)

    def _sample_future(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self.offset) + 1
        obs = episode["observation"][idx - 1]
        action = ["action"][idx]
        next_obs = episode["observation"][idx + self.offset]
        reward = episode["reward"][idx]
        discount = episode["discount"][idx] * self._discount
        return (obs, action, reward, discount, next_obs)
    
                
    def __iter__(self):
        while True:
            if self.distill:
                yield self._sample()
            elif self.goal:
                yield self._sample_goal()
            elif self.vae:
                yield self._sample_future()
            else:
                yield self._sample()

    def parse_dataset(self, start_ind=0, end_ind=-1):
        states = []
        actions = []
        futures = []
        for eps_fn in tqdm.tqdm(self._episode_fns[start_ind:end_ind]):
            episode = self._episodes[eps_fn]
            offsets = [25, 50, 75, 100, 100]
            ep_len = next(iter(episode.values())).shape[0] - 1
            for idx in range(ep_len - 100):
                ys = []
                states.append(episode["observation"][idx - 1][None])
                actions.append(episode["action"][idx][None])
                for off in offsets:
                    ys.append(episode["observation"][idx + off][:,None])
                futures.append(np.concatenate(ys,1)[None])
        return (
            np.concatenate(states, 0),
            np.concatenate(actions, 0),
            np.concatenate(futures, 0),
        )


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_buffer(
    env,
    replay_dir,
    max_size,
    batch_size,
    num_workers,
    discount,
    offset=100,
    goal=False,
    vae=False,
    relabel=False
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
    iterable._load(relabel=relabel)

    return iterable


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
    distill=False,
    relabel=False
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
        distill=distill,
        relabel=relabel
    )
    iterable._load()

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


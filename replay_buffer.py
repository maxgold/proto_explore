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
from path_collector import PathBuilder


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
    #print('original reward ', episode["reward"]) 
    for i in range(states.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(states[i])
            ## don't we need to step before getting the new state? 
        reward = env.task.get_reward(env.physics)
        #print('new reward', reward) 
        reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
        rewards.append(reward)
    episode["reward"] = np.array(rewards, dtype=reward_spec.dtype)
    #print('new goal from env.task.get_reward', episode["achieved_goal"])
    return episode


class OfflineReplayBuffer(IterableDataset):


    def __init__(self, env, replay_dir, max_size, num_workers, discount,goal, offset=100, offset_schedule=None):

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
        self.vae = False
        self.threshold = 0.001


    def _load(self, relable=True):
        print("Labeling data...")
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"))
        # for eps_fn in tqdm.tqdm(eps_fns):
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
        return (obs, action, reward, discount, next_obs)

    def _sample_goal(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self.offset) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx]
        goal = episode["observation"][idx + self.offset]
        # goal = np.random.rand(2)
        #control_reward = rewards.tolerance(
        #    action, margin=1, value_at_margin=0, sigmoid="quadratic"
        #).mean()
        #small_control = (control_reward + 4) / 5
        #reward = np.linalg.norm(goal[:2] - next_obs[:2]) * small_control
        
        dist = np.linalg.norm(goal - next_obs)
        reward = np.zeros((1,1))
        reward[np.where(dist > self.threshold)] = -1
        reward = reward.reshape(-1)
        discount = np.ones_like(episode["discount"][idx])
        
        future = episode['observation'][np.random.randint(0, episode_len(episode))]
        return (obs, action, reward, discount, next_obs, goal, future)



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

    def __iter__(self):
        while True:
            if self.goal:
                yield self._sample_goal()
            elif self.vae:
                yield self._sample_future()
            else:
                yield self._sample()


#             for i in range(1,100):
#                 self.offset=i
#                 for ix in range(100000):
#                     if ix==1:
#                         print('sample', self._sample(), 'offset', i)
#                     yield self._sample()
# #             yield self._sample_future()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(
    env, replay_dir, max_size, batch_size, num_workers, discount, offset=100, goal=False
):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = OfflineReplayBuffer(
        env, replay_dir, max_size_per_worker, num_workers, discount, offset, goal)
    iterable._load(False)

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader

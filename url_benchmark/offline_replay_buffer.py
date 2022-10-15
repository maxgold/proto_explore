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
        r = 1
    else:
        upper = 0.015
        margin = 0.1
        scale = np.sqrt(-2 * np.log(0.1))
        x = (dist_to_target - upper) / margin
        r = np.exp(-0.5 * (x * scale) ** 2)
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

    def add(self, time_step, meta):
        for key, value in meta.items():
            self._current_episode[key].append(value)
        for spec in self._data_specs:
            value = time_step[spec.name]
            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)
        if time_step.last():
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            for spec in self._meta_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
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


GOAL_ARRAY = np.array([[-0.15, 0.15], [-0.15, -0.15], [0.15, -0.15], [0.15, 0.15]])


class OfflineReplayBuffer(IterableDataset):
    def __init__(
        self,
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

    def _load(self, relabel=True):
        print("Labeling data...")
        relabel = False
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"))
        # for eps_fn in tqdm.tqdm(eps_fns):
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

    def _sample_goal(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition

        idx = np.random.randint(0, episode_len(episode) - self.offset) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx]
        goal = episode["observation"][idx + self.offset][:2]
        rewards = []
        cand_goals = np.random.uniform(-.2,.2, size=(50,2))
        cand_goals = cand_goals[np.abs(cand_goals[:,0])>.05]
        cand_goals = cand_goals[np.abs(cand_goals[:,1])>.05]
        cand_goals = cand_goals[:4]
        for goal in cand_goals:
            rewards.append(my_reward(action, next_obs, goal))
        #rewards.append(my_reward(action, next_obs, GOAL_ARRAY[0]))
        discount = np.ones_like(episode["discount"][idx])
        obs = np.tile(obs, (4, 1))
        action = np.tile(action, (4, 1))
        discount = np.tile(discount, (4, 1))
        next_obs = np.tile(next_obs, (4, 1))
        reward = np.array(rewards)

        return (obs, action, reward, discount, next_obs, cand_goals)

    def _sample_gcsl(self):
        episode = self._sample_episode()
        h = random.sample(range(1000),1)[0]
        idx = np.random.randint(0, episode_len(episode) - h) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        goal = episode["observation"][idx + h][:2]
        horizon = np.array(int(h/10))
        #obses.append(obs[None])
        #goals.append(goal[None])
        #actions.append(action[None])
        #horizons2.append(np.array(h)[None])

#        for h in horizons:
#            idx = np.random.randint(0, episode_len(episode) - h) + 1
#            obs = episode["observation"][idx - 1]
#            action = episode["action"][idx]
#            goal = episode["observation"][idx + h][:2]
#            obses.append(obs[None])
#            goals.append(goal[None])
#            actions.append(action[None])
#            horizons2.append(np.array(h)[None])
#        goals = np.concatenate(goals)
#        obses = np.concatenate(obses)
#        actions = np.concatenate(actions)
#        horizons2 = np.array(horizons2)
        
        return (obs, action, goal, horizon)

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
            if self.goal:
                #yield self._sample_goal()
                yield self._sample_gcsl()
                #yield self._sample_goal(), self._sample_gcsl()
                #yield np.zeros(10), self._sample_gcsl()
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

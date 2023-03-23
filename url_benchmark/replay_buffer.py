import datetime
import io
import random
import traceback
import copy
from collections import defaultdict
import tqdm
import re
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from dm_control.utils import rewards
from itertools import chain
from pathlib import Path
import glob


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1

def save_episode(episode, fn):
#    deepdish.io.save(fn, episode)
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn, eval=False):
    with fn.open("rb") as f:
        episode = np.load(f)
        if eval:
            if 'goal_state' in episode.keys():
                keys = ['state', 'reward', 'action', 'goal_state']
            
                episode = {k: episode[k] for k in keys}
            else:
                keys = ['state', 'reward', 'action']
                episode = {k: episode[k] for k in keys}
        else:
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


def my_reward(action, next_obs, goal, ant=False):
    # this is optimized for speed ...
    tmp = 1 - action**2
    if ant is False:
        control_reward = max(min(tmp[0], 1), 0) / 2
        control_reward += max(min(tmp[1], 1), 0) / 2
    else:
        control_reward = 0
        for i in range(action.shape[0]):
            control_reward += max(min(tmp[i], 1), 0) / action.shape[0]

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
    def __init__(self, data_specs, meta_specs, replay_dir, visitation_matrix_size=60, visitation_limit=.29, obs_spec_keys=None, act_spec_keys=None):
        self._data_specs = data_specs
        self._meta_specs = meta_specs
        self._replay_dir = replay_dir
        last = str(replay_dir).split('/')[-1]
        self._replay_dir2 = replay_dir.parent / last / "buffer_copy"
        replay_dir.mkdir(exist_ok=True)
        (replay_dir.parent / last / "buffer_copy").mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._current_episode_goal = defaultdict(list)
        self._preload()
        self.state_visitation_proto = np.zeros((visitation_matrix_size,visitation_matrix_size))
        self.state_visitation_proto_pct = np.zeros((20,20))
        self.state_visitation_gc = np.zeros((visitation_matrix_size,visitation_matrix_size))
        self.state_visitation_gc_pct = np.zeros((20,20))
        self.reward_matrix = np.zeros((visitation_matrix_size,visitation_matrix_size))
        self.goal_state_matrix = np.zeros((visitation_matrix_size,visitation_matrix_size))
        self.visitation_limit = visitation_limit*100
        if obs_spec_keys is not None:
            self.obs_spec_keys = obs_spec_keys
        if act_spec_keys is not None:
            self.act_spec_keys = act_spec_keys
    def __len__(self):
        return self._num_transitions

    def add(self, time_step, state=None, meta=None,pixels=False, last=False, pmm=True, action=None):
        for key, value in meta.items():
            self._current_episode[key].append(value)
        for spec in self._data_specs:
            if spec.name == 'observation' and pixels:
                value = time_step[spec.name]    
                self._current_episode['observation'].append(value['pixels'])
                if state is not None:
                    self._current_episode['state'].append(state)
                else:
                    self._current_episode['state'].append(value['observations'])

                if pmm:
                    tmp_state = value['observations']*100
                    idx_x = int(tmp_state[0])+self.visitation_limit
                    idx_y = int(tmp_state[1])+self.visitation_limit
                    self.state_visitation_proto[idx_x,idx_y]+=1
                
                    # tmp_state = tmp_state/3
                    # idx_x = int(tmp_state[0])+9
                    # idx_y = int(tmp_state[1])+9
                    # self.state_visitation_proto_pct[idx_x,idx_y]+=1
            elif spec.name == 'action' and action is not None:
                value = action
                if np.isscalar(value):
                    value = np.full(spec.shape, value, spec.dtype)
                assert spec.shape == value.shape and spec.dtype == value.dtype
                self._current_episode[spec.name].append(value)
            else:
                
                value = time_step[spec.name]
                if np.isscalar(value):
                    value = np.full(spec.shape, value, spec.dtype)
                assert spec.shape == value.shape and spec.dtype == value.dtype
                self._current_episode[spec.name].append(value)
                    
        if time_step.last() or last:
            episode = dict()
            for spec in self._data_specs:
                if spec.name == 'observation' and pixels:
                    value1 = self._current_episode['observation']
                    value2 = self._current_episode['state']
                    episode['observation'] = np.array(value1, spec.dtype)
                    episode['state'] = np.array(value2, 'float32')
                    
                else:
                    value = self._current_episode[spec.name]
                    episode[spec.name] = np.array(value, spec.dtype)
            
            for spec in self._meta_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            print('episode', episode.values())
            self._store_episode(episode, actor1=False)
            print('storing episode, no goal')
    
    def add_gym(self, obs, action, meta=None, last=False):

        for spec in self._data_specs:
            if spec.name == 'image':
                value = obs['image'].transpose(2,0,1)

                #saving stats for heatmap later 
                tmp_state = obs['walker/world_zaxis']*100
                # print('check the limits of the x, y coordinates')
                idx_x = int(tmp_state[0])+self.visitation_limit
                idx_y = int(tmp_state[1])+self.visitation_limit
                self.state_visitation_proto[idx_x,idx_y]+=1
            
            elif spec.name == 'reward':
                value = obs[spec.name]
            elif spec.name in self.obs_spec_keys:
                value = obs[spec.name]
            elif spec.name in self.act_spec_keys:
                value = action[spec.name]

            value = np.array(value, spec.dtype)

            if np.isscalar(value):
                value = np.full(spec.shape, value, spec.dtype)
                
            assert spec.shape == value.shape and spec.dtype == value.dtype
            self._current_episode[spec.name].append(value)

        if last:
            episode = dict()
            for spec in self._data_specs:
                if spec.name not in (self.obs_spec_keys + self.act_spec_keys + ['reward']):
                    continue
                else:
                    value = self._current_episode[spec.name]
                    episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list)
            self._store_episode(episode, actor1=False)
            print('storing episode, no goal, gym')
            # next(iter(episode.values())).shape[0] - 1


    def add_goal(self, time_step, meta, goal, time_step_no_goal=False,goal_state=False,pixels=False, last=False, asym=False):
        for key, value in meta.items():
            self._current_episode_goal[key].append(value)
        for spec in self._data_specs:
            if spec.name == 'observation' and pixels:
                value = time_step_no_goal[spec.name]
                self._current_episode_goal['observation'].append(value['pixels'])
                self._current_episode_goal['state'].append(value['observations'])
                
                
                tmp_state = value['observations']*100
                idx_x = int(tmp_state[0])+self.visitation_limit
                idx_y = int(tmp_state[1])+self.visitation_limit
                self.state_visitation_gc[idx_x,idx_y]+=1
                
                # tmp_state = tmp_state/3
                # idx_x = int(tmp_state[0])+9
                # idx_y = int(tmp_state[1])+9
                # self.state_visitation_gc_pct[idx_x,idx_y]+=1
                
            else:
                value = time_step[spec.name]
                if np.isscalar(value):
                    value = np.full(spec.shape, value, spec.dtype)
                assert spec.shape == value.shape and spec.dtype == value.dtype
                self._current_episode_goal[spec.name].append(value)
                
                if spec.name == 'reward' and pixels:
                    value = time_step['observation']
                    tmp_state = value['observations']*100
                    idx_x = int(tmp_state[0])+self.visitation_limit
                    idx_y = int(tmp_state[1])+self.visitation_limit
                    self.reward_matrix[idx_x,idx_y]+=time_step['reward']
                
        if pixels and asym==False:
            goal = np.transpose(goal, (2,0,1))
            self._current_episode_goal['goal_state'].append(goal_state)
            idx_x = int(goal_state[0]*100)+self.visitation_limit
            idx_y = int(goal_state[1]*100)+self.visitation_limit
            self.goal_state_matrix[idx_x,idx_y]+=1
            
        elif pixels and asym:
            self._current_episode_goal['goal_state'].append(goal_state)
            idx_x = int(goal_state[0]*100)+self.visitation_limit
            idx_y = int(goal_state[1]*100)+self.visitation_limit
            self.goal_state_matrix[idx_x,idx_y]+=1
        self._current_episode_goal['goal'].append(goal)

        if time_step.last() or last:
            # print('replay last')
            episode = dict()
            for spec in self._data_specs:
                if spec.name == 'observation' and pixels:
                    value1 = self._current_episode_goal['observation']
                    value2 = self._current_episode_goal['state']
                    episode['observation'] = np.array(value1, spec.dtype)
                    episode['state'] = np.array(value2, 'float32')
                else:
                    value = self._current_episode_goal[spec.name]
                    episode[spec.name] = np.array(value, spec.dtype)

            for spec in self._meta_specs:
                value = self._current_episode_goal[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            value = self._current_episode_goal['goal']
            
            if pixels and asym==False:
                episode['goal'] = np.array(value).astype(int) 
            else:
                episode['goal'] = np.array(value, np.float64)
            
            if pixels:
                value = self._current_episode_goal['goal_state']
                episode['goal_state'] = np.array(value, np.float64)
            self._current_episode_goal = defaultdict(list)
            self._store_episode(episode, actor1=True)
            # print('storing episode, w/ goal')
            
    def add_goal_general(self, time_step, state, meta, goal, goal_state, time_step_no_goal, pixels=False, last=False, asym=False, expert=False):
        #assert goal.shape[0]==9 and goal.shape[1]==84 and goal.shape[2]==84
        if time_step_no_goal is not None:
            pmm=True
        else:
            pmm=False
        
        if expert is False:
            for key, value in meta.items():
                self._current_episode_goal[key].append(value)
            
        for spec in self._data_specs:
            if spec.name == 'observation' and pixels:
                if pmm:
                    value = time_step_no_goal[spec.name]
                    self._current_episode_goal['observation'].append(value['pixels'])
                    self._current_episode_goal['state'].append(value['observations'])


                    tmp_state = value['observations']*100
                    idx_x = int(tmp_state[0])+self.visitation_limit
                    idx_y = int(tmp_state[1])+self.visitation_limit
                    self.state_visitation_gc[idx_x,idx_y]+=1

                    #tmp_state = tmp_state/3
                    #idx_x = int(tmp_state[0])+9
                    #idx_y = int(tmp_state[1])+9
                    #self.state_visitation_gc_pct[idx_x,idx_y]+=1
                else:
                    value = time_step[spec.name]
                    self._current_episode_goal['observation'].append(value['pixels'])
                    self._current_episode_goal['state'].append(state)
                
            else:
                value = time_step[spec.name]
                
                if np.isscalar(value):
                    value = np.full(spec.shape, value, spec.dtype)
                assert spec.shape == value.shape and spec.dtype == value.dtype
                
                self._current_episode_goal[spec.name].append(value)
                
                if spec.name == 'reward' and pixels and pmm:
                    assert time_step['reward']>=0.
                    value = time_step['observation']
                    tmp_state = value['observations']*100
                    idx_x = int(tmp_state[0])+self.visitation_limit
                    idx_y = int(tmp_state[1])+self.visitation_limit
                    self.reward_matrix[idx_x,idx_y]+=time_step['reward']
                
        if pixels and asym==False and pmm:
            
            #goal = np.transpose(goal, (2,0,1))
            self._current_episode_goal['goal_state'].append(goal_state)
            idx_x = int(goal_state[0]*100)+self.visitation_limit
            idx_y = int(goal_state[1]*100)+self.visitation_limit
            self.goal_state_matrix[idx_x,idx_y]+=1
            
        elif pixels and asym==False:
            self._current_episode_goal['goal_state'].append(goal_state)
            
        self._current_episode_goal['goal'].append(goal)

        if time_step.last() or last:
            
            # print('replay last')
            episode = dict()
            
            for spec in self._data_specs:
                
                if spec.name == 'observation' and pixels:
                    value1 = self._current_episode_goal['observation']
                    value2 = self._current_episode_goal['state']
                    episode['observation'] = np.array(value1, spec.dtype)
                    episode['state'] = np.array(value2, 'float32')
                else:
                    value = self._current_episode_goal[spec.name]
                    episode[spec.name] = np.array(value, spec.dtype)
            if expert is False:
                for spec in self._meta_specs:
                    value = self._current_episode_goal[spec.name]
                    episode[spec.name] = np.array(value, spec.dtype)
                
            value = self._current_episode_goal['goal']
            if pixels and asym==False:
                episode['goal'] = np.array(value).astype(int)
                
            else:
                episode['goal'] = np.array(value, np.float64)
            if np.any(episode['goal']<0):
                import IPython as ipy; ipy.embed(colors='neutral')
            if pixels:
                value = self._current_episode_goal['goal_state']
                episode['goal_state'] = np.array(value, np.float64)
                
            self._current_episode_goal = defaultdict(list)
            self._store_episode(episode, actor1=True)
            # print('storing episode, w/ goal, general')
            
            
    def add_proto_goal(self, time_step, z, meta, goal, reward, last=False, goal_state=None, neg_reward=False, pmm=True):
        for key, value in meta.items():
            self._current_episode_goal[key].append(value)
        for spec in self._data_specs:
            if spec.name=='observation':
                              #continue updating code here 
                value1 = np.array([z]).reshape((-1,))
                self._current_episode_goal['observation'].append(value1)
                value2 = time_step[spec.name]
                self._current_episode_goal['state'].append(value2['observations'])
                
                if pmm:
                    tmp_state = value2['observations']*100
                    idx_x = int(tmp_state[0])+self.visitation_limit
                    idx_y = int(tmp_state[1])+self.visitation_limit
                    self.state_visitation_gc[idx_x,idx_y]+=1
                    # tmp_state = tmp_state/3
                    # idx_x = int(tmp_state[0])+9
                    # idx_y = int(tmp_state[1])+9
                    # self.state_visitation_gc_pct[idx_x,idx_y]+=1

            elif spec.name=='reward' and pmm:
                
                value = np.array([reward]).reshape((-1,))*2
                self._current_episode_goal['reward'].append(value)

                value = time_step['observation']
                tmp_state = value['observations']*100
                idx_x = int(tmp_state[0])+self.visitation_limit
                idx_y = int(tmp_state[1])+self.visitation_limit
                if neg_reward:
                    self.reward_matrix[idx_x,idx_y]+=reward+1
                else:
                    self.reward_matrix[idx_x,idx_y]+=reward
            else:
                value = time_step[spec.name]
                if np.isscalar(value):
                    value = np.full(spec.shape, value, spec.dtype)
                assert spec.shape == value.shape and spec.dtype == value.dtype
                self._current_episode_goal[spec.name].append(value)
        
        goal = np.array([goal]).reshape((-1,))
        self._current_episode_goal['goal'].append(goal)

        if goal_state is not None and pmm:
            self._current_episode_goal['goal_state'].append(goal_state)
            idx_x = int(goal_state[0]*100)+self.visitation_limit
            idx_y = int(goal_state[1]*100)+self.visitation_limit
            self.goal_state_matrix[idx_x,idx_y]+=1 

        if time_step.last() or last:
            # print('replay last')
            episode = dict()
            for spec in self._data_specs:
                if spec.name=='observation':
                    #continue updating code here 
                    value1 = self._current_episode_goal['observation']
                    value2 = self._current_episode_goal['state']
                    episode['observation'] = np.array(value1, 'float32')
                    episode['state'] = np.array(value2, 'float32')
                elif spec.name=='reward':
                    value = self._current_episode_goal['reward']
                    episode['reward']= np.array(value, 'float32')
                else:
                    value = self._current_episode_goal[spec.name]
                    episode[spec.name] = np.array(value, spec.dtype)

            for spec in self._meta_specs:
                value = self._current_episode_goal[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            
            value = self._current_episode_goal['goal']
            episode['goal'] = np.array(value)
            
            if goal_state is not None:
                value = self._current_episode_goal['goal_state']
                episode['goal_state'] = np.array(value, np.float64)

            self._current_episode_goal = defaultdict(list)
            self._store_episode(episode, actor1=True)
            # print('storing episode, w/ goal, proto')
         

    def add_q(self, time_step, meta, q, task):
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

    def _store_episode(self, episode, actor1=False):
        eps_idx = self._num_episodes
        eps_len = episode_len(episode)
        self._num_episodes += 1
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S')
        eps_fn = f"{ts}_{eps_idx}_{eps_len}.npz"
        print('storing', eps_fn)
        save_episode(episode, self._replay_dir / eps_fn)
        save_episode(episode, self._replay_dir2 / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(
        self,
        storage,
        storage2,
        max_size,
        num_workers,
        nstep,
        discount,
        goal,
        hybrid,
        obs_type,
        fetch_every,
        save_snapshot,
        hybrid_pct,
        actor1=False,
        replay_dir2=False,
        goal_proto=False,
        agent=None,
        neg_reward=False,
        sl=False,
        asym=False,
        loss=False,
        test=False,
        tile=1,
        pmm=True,
        obs_shape=4,
        general=False,
        inv=False,
        goal_offset=1,
        gym=False):
        self._storage = storage
        self._storage2 = storage2
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        print('nstep', nstep)
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot
        self.goal = goal
        self.hybrid = hybrid
        self.actor1 = actor1
        self.hybrid_pct = hybrid_pct
        self.count=0
        self.count1=0
        self.count2=0
        self.last=-1
        self.iz=1
        self._replay_dir2=replay_dir2
        self.goal_proto = goal_proto
        self.agent = agent
        self.neg_reward = neg_reward
        self.sl = sl
        self.asym = asym
        self.loss = loss
        self.test = test
        self.tile = tile
        self.pmm = pmm
        self.index = obs_shape//2
        self.general = general
        self.inv = inv
        self.goal_offset = goal_offset
        self.gym = gym
        if obs_type == 'pixels':
            self.pixels = True
        else:
            self.pixels = False

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
        #if hyperparameter: second=True, hybrid_pct=x
        #print('eps', self._episode_fns)
        if self._storage2 and self.hybrid_pct!=0:
        
            eps_fns1 = sorted(self._storage._replay_dir.glob("*.npz"), reverse=True)
            tmp_fns = sorted(self._replay_dir2.glob("*.npz"))
            #print('tmp', tmp_fns)
            tmp_fns_=[]
            tmp_fns2 = []

            for x in tmp_fns:
                tmp_fns_.append(str(x))
                tmp_fns2.append(x)
            #if self.model_step:
            #    eps_fns2 = [tmp_fns2[ix] for ix,x in enumerate(tmp_fns_) if (int(re.findall('\d+', x)[-2]) < self.model_step)]
            #else:
            eps_fns2 = tmp_fns

            #np.random.shuffle(eps_fns2)
            fetched_size = 0
            for eps_fn1 in eps_fns1:
                #print('count1', self.count1)
                #if self.count1 < 10-self.hybrid_pct*10:
                eps_idx, eps_len = [int(x) for x in eps_fn1.stem.split("_")[1:]]
                
                if eps_idx % self._num_workers != worker_id:
                    continue
                if eps_fn1 in self._episodes.keys():
                    break
                if fetched_size + eps_len > self._max_size:
                    break
                fetched_size += eps_len
                if not self._store_episode(eps_fn1):
                    break
                #self.count1 += 1
                
            for ix, eps_fn2 in enumerate(eps_fns2):

                #if self.count2 < self.hybrid_pct*10:
                print('eps2', eps_fn2)
                #if ix!=self.last+1:
                #    continue
                #else:
                eps_idx, eps_len = [int(x) for x in eps_fn2.stem.split("_")[1:]]
                if eps_idx % self._num_workers != worker_id:
                    continue
                if eps_fn2 in self._episodes.keys():
                    break
                if fetched_size + eps_len > self._max_size:
                    break
                fetched_size += eps_len
                if not self._store_episode(eps_fn2):
                    break
                #self.count2 += 1
                #self.last=ix

                #else:
                #    break
            
            #print('final count1',self.count1)
            #print('final count2', self.count2)
            
            #if self.count1 == (10-self.hybrid_pct*10-1) and self.count2 == (self.hybrid_pct*10-1):
            #    print('reset')
            #    print('reset count1', self.count1)
            #    print('reset count2', self.count2)
            #    self.count1 = 0
            #    self.count2 = 0

        #if hyperparameter: second=False
        else:
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
                    #print('break')
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
        obs_state =  episode["state"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]
        next_obs_state = episode["state"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        
        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount
         
        if self.goal:
            goal = episode["goal"][idx-1]
            if self.pixels and self.goal_proto==False:
                goal = np.tile(goal,(self.tile,1,1))
            return (obs, action, reward, discount, next_obs, goal, *meta)

        elif self.loss:
            
            obs_state =  episode["state"][idx - 1]
            next_obs_state = episode["state"][idx + self._nstep - 1]
            
            episode = self._sample_episode()
            idx = np.random.randint(0, episode_len(episode))
            rand_obs = episode['observation'][idx - 1]
            return (obs, obs_state, action, reward, discount, next_obs, next_obs_state, rand_obs, *meta)	
        else:
            return (obs, action, reward, discount, next_obs, next_obs_state, *meta)
    
    def _sample_gym(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1

        # if self.prio_starts:
        # lower -= int(self.chunk * self.prio_starts)
        # if self.prio_ends:
        # upper += int(self.chunk * self.prio_ends)

        state = []
        next_state = []
        #if need to change obs_spec_keys that are used, then use original implemnetation 
        #from https://github.com/danijar/director/blob/a6a649efedc25a9d194155682bf39163fba4b5a2/embodied/agents/director/nets.py#L172

        obs = episode['image'][idx - 1]
        next_obs = episode['image'][idx + self._nstep - 1]
        for key in self._storage.obs_spec_keys:
            if key == 'image':
                obs = episode[key][idx - 1]
                next_obs = episode[key][idx + self._nstep - 1]
            else:
                state_ = episode[key][idx - 1]
                next_state_ = episode[key][idx + self._nstep - 1]
                state.append(state_)
                next_state.append(next_state_)
        state = np.concatenate(state, axis=-1)
        next_state = np.concatenate(next_state, axis=-1)
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["reward"][idx])
        action = episode["action"][idx]

        #joint states
        # state = {k: episode[k][idx-1] for k in self._storage.obs_spec_keys()}
        # chunk['is_first'] = np.zeros(len(chunk['action']), bool)
        # chunk['is_first'][0] = True

        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= self._discount * self._discount

        return (obs, state, action, reward, discount, next_obs, next_state)

        
    def _sample_inv(self):
        
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
            
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        
        # add +1 for the first dummy transition
        offset = np.random.randint(self.goal_offset)
        if (episode_len(episode) - self._nstep - offset + 1) <=0:
            offset = np.random.randint(episode_len(episode) - self._nstep +1)
        idx = np.random.randint(0, episode_len(episode) - self._nstep - offset + 1) + 1
        meta = []
        
        for spec in self._storage._meta_specs:
            meta.append(episode[spec.name][idx - 1])

        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]
        next_obs_state = episode["state"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        obs_state = episode["state"][idx - 1]
        #idx_goal = np.random.randint(idx + self._nstep - 1,episode_len(episode))    
        idx_goal = idx + self._nstep + offset - 1
        goal = episode["observation"][idx_goal]

        if (goal.shape[0]//3)!=self.tile:
            goal = np.tile(goal,(self.tile,1,1))
        else:
            goal = goal[:self.tile*3,:,:]
        
        goal_state = episode["state"][idx_goal,:2]
        
        for i in range(self._nstep):
            
                if self.pmm:
                    step_reward = my_reward(episode["action"][idx+i],episode["state"][idx+i] , goal_state[:2])*2
                else:
                    step_reward = -np.linalg.norm(episode["state"][idx+i][:self.index]-goal_state[:self.index])
                    
                reward += discount * step_reward
                discount *= episode["discount"][idx+i] * self._discount

        return (obs, obs_state, action, reward, discount, next_obs, next_obs_state, goal, goal_state, *meta)


    def _sample_inv_gym(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
            
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        
        # add +1 for the first dummy transition
        offset = np.random.randint(self.goal_offset)
        if (episode_len(episode) - self._nstep - offset + 1) <=0:
            offset = np.random.randint(episode_len(episode) - self._nstep +1)
        idx = np.random.randint(0, episode_len(episode) - self._nstep - offset + 1) + 1

        state = []
        next_state = []

        obs = episode['image'][idx - 1]
        next_obs = episode['image'][idx + self._nstep - 1]
        for key in self._storage.obs_spec_keys:
            if key == 'image':
                obs = episode[key][idx - 1]
                next_obs = episode[key][idx + self._nstep - 1]
            else:
                state_ = episode[key][idx - 1]
                next_state_ = episode[key][idx + self._nstep - 1]
                state.append(state_)
                next_state.append(next_state_)

        state = np.concatenate(state, axis=-1)
        next_state = np.concatenate(next_state, axis=-1)
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["reward"][idx])
        action = episode["action"][idx]

        idx_goal = idx + self._nstep + offset - 1
        goal = episode['image'][idx_goal]
        goal_state = episode['walker/world_zaxis'][idx_goal]

        if (goal.shape[0]//3)!=self.tile:
            goal = np.tile(goal,(self.tile,1,1))
        else:
            goal = goal[:self.tile*3,:,:]
 
        for i in range(self._nstep):
                step_reward = my_reward(episode["action"][idx+i], episode['walker/world_zaxis'][idx+i][:2] , goal_state[:2])*2
                # else:
                #     step_reward = -np.linalg.norm(episode["state"][idx+i][:self.index]-goal_state[:self.index])
                reward += discount * step_reward
                discount *= self._discount * self._discount

        return (obs, obs_state, action, reward, discount, next_obs, next_obs_state, goal, goal_state, *meta)


    def _append(self):
        #add all, goal or no goal trajectories, used for sampling goals
        final = []
        episode_fns = self._episode_fns1 + self._episode_fns2
        for eps_fn in episode_fns:
            final.append(self._episodes[eps_fn]['observation'])
        #import IPython as ipy; ipy.embed(colors='neutral')
        if len(final)>0:
            final = torch.cat(final)
            return final
        else:
            print('nothing in buffer yet')
          
            return ''
        
    def parse_dataset(self, start_ind=0, end_ind=-1,goal_state=None, proto_goal=False):
        states = []
        actions = []
        rewards = []
        episode_name = []
        index = []
        proto=[]
        if goal_state:
            goal_states=[]
        if len(self._episode_fns)>0:
            for eps_fn in tqdm.tqdm(self._episode_fns[start_ind:end_ind]):
                episode = self._episodes[eps_fn]
                ep_len = next(iter(episode.values())).shape[0] - 1
                for idx in range(ep_len):
                    
                    if self.pixels:
                        states.append(episode["state"][idx - 1][None])
                    else:
                        states.append(episode["observation"][idx - 1][None])
                    actions.append(episode["action"][idx][None])
                    rewards.append(episode["reward"][idx][None])
                    episode_name.append(str(eps_fn))
                    
                    if proto_goal:
                        proto.append(episode["observation"][idx - 1][None])
                    
                    index.append(np.array([idx]))
                    
                    if goal_state:
                        goal_states.append((episode["goal_state"][idx][None]))
            
            if goal_state:
      
                return (np.concatenate(states,0),
                        np.concatenate(actions, 0),
                        np.concatenate(rewards, 0),
                        np.concatenate(goal_states, 0),
                        episode_name,
                        index
                        )
            elif proto:
                return (np.concatenate(states,0),
                        np.concatenate(proto,0),
                        np.concatenate(actions, 0),
                        np.concatenate(rewards, 0),
                        episode_name,
                        index
                        )
            else:
                print('states', len(states))
                return (np.concatenate(states,0),
                        np.concatenate(actions, 0),
                        np.concatenate(rewards, 0),
                        episode_name,
                        index
                        )
        else:
            return ('', '')

    def __iter__(self):
        while True:
            if self.inv:
                if self.gym is False:
                    yield self._sample_inv()
                else:
                    yield self._sample_inv_gym()
            else:
                if self.gym is False:
                    yield self._sample()
                else:
                    yield self._sample_gym()


class OfflineReplayBuffer(IterableDataset):
    def __init__(self,
        replay_dir,
        max_size,
        num_workers,
        discount,
        offset=100,
        offset_schedule=None,
        random_goal=False,
        goal=False,
        replay_dir2=False,
        obs_type='state',
        hybrid=False,
        hybrid_pct=0,
        offline=False,
        nstep=1,
        load_every=10000,
        eval=False,
        load_once=False,
        inv=False,
        goal_offset=1,
        pmm=True,
        model_step=None,
        model_step_lb=None,
        reverse=True,
        gym=False,
        obs_spec_keys=None,
        tile=1,):

        self._replay_dir = replay_dir
        self._replay_dir2 = replay_dir2
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
        self.goal_array = []
        self._goal_array = False
        self.obs = []
        self.offline = offline
        
        self.hybrid = hybrid
        if self.offline:
            self.hybrid_pct = 1
        else:
            
            self.hybrid_pct = hybrid_pct
        self._nstep=nstep
        self.count=0
        self.iz=1
        self._load_every = load_every
        self._samples_since_last_load = load_every
        self.load_once=load_once
        self.switch=False
        self.inv=inv
        self.goal_offset=goal_offset
        self.gym=gym
        self.obs_spec_keys = obs_spec_keys
        self.tile = tile
        if model_step is not None:
            self.model_step = int(model_step//500)
        else:
            self.model_step = None
        if model_step_lb is not None:
            self.model_step_lb = int(model_step_lb//500)
        else:
            self.model_step_lb = 0
        print('goal offset', goal_offset)
        print('model step', self.model_step)
        print('model step lb', self.model_step_lb)

        if obs_type == 'pixels':
            self.pixels = True      
        else:
            self.pixels = False
        self.eval = eval
        self.pmm = pmm
        self.reverse = reverse

    def _load(self, relabel=False):
        if self._samples_since_last_load < self._load_every and len(self._episode_fns)!=0:
            print('samples since last load', self._samples_since_last_load)
            return

        self._samples_since_last_load = 0
        print('loading offline data')

        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:   
            worker_id = 0

        if self.reverse:
            eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
        else:
            eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=False)
        tmp_fns_=[]
        tmp_fns2 = []

        for x in eps_fns:
            tmp_fns_.append(str(x))
            tmp_fns2.append(x)
                
        if self.model_step is not None:
            eps_fns = [tmp_fns2[ix] for ix,x in enumerate(tmp_fns_) if (self.model_step_lb < int(re.findall('\d+', x)[-2]) < self.model_step)]
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
            print('append', eps_fn)
            self._episodes[eps_fn] = episode
            self._size += episode_len(episode)

    def _sample_episode(self):
        if not self._loaded or len(self._episode_fns)==0:
            self._load()
            self._loaded = True
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _relabel_reward(self, episode):
        return relabel_episode(self._env, episode)
    
    def _sample_inv(self):
        if self._load_every!=0:
            try:
                self._load()
            except:
                traceback.print_exc()
            self._samples_since_last_load += 1
        episode = self._sample_episode()
        
        # add +1 for the first dummy transition
        offset = np.random.randint(self.goal_offset)
        if (episode_len(episode) - self._nstep - offset + 1) <=0:
            offset = np.random.randint(episode_len(episode) - self._nstep +1)

        idx = np.random.randint(0, episode_len(episode) - self._nstep - offset + 1) + 1

        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]
        next_obs_state = episode["state"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        obs_state = episode["state"][idx - 1]
        #idx_goal = np.random.randint(idx + self._nstep - 1,episode_len(episode))    
        idx_goal = idx + self._nstep + offset - 1
        goal = episode["observation"][idx_goal]
        goal_state = episode["state"][idx_goal,:2]
        
        for i in range(self._nstep):
            if self.pmm:
                step_reward = my_reward(episode["action"][idx+i],episode["state"][idx+i] , goal_state[:2])*2
            else:
                step_reward = -np.linalg.norm(episode["state"][idx+i][:self.index]-goal_state[:self.index]) 
            reward += discount * step_reward
            discount *= episode["discount"][idx+i] * self._discount         
        return (obs, obs_state, action, reward, discount, next_obs, next_obs_state, goal, goal_state)
    
    def _sample_inv_gym(self):
        if self._load_every!=0:
            try:
                self._load()
            except:
                traceback.print_exc()
            self._samples_since_last_load += 1
        episode = self._sample_episode()
        
        # add +1 for the first dummy transition
        offset = np.random.randint(self.goal_offset)
        if (episode_len(episode) - self._nstep - offset + 1) <=0:
            offset = np.random.randint(episode_len(episode) - self._nstep +1)
        idx = np.random.randint(0, episode_len(episode) - self._nstep - offset + 1) + 1

        state = []
        next_state = []

        obs = episode['image'][idx - 1]
        next_obs = episode['image'][idx + self._nstep - 1]
        for key in self.obs_spec_keys:
            if key == 'image':
                obs = episode[key][idx - 1]
                next_obs = episode[key][idx + self._nstep - 1]
            else:
                state_ = episode[key][idx - 1]
                next_state_ = episode[key][idx + self._nstep - 1]
                state.append(state_)
                next_state.append(next_state_)

        obs_state = np.concatenate(state, axis=-1)
        next_obs_state = np.concatenate(next_state, axis=-1)
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["reward"][idx])
        action = episode["action"][idx]

        idx_goal = idx + self._nstep + offset - 1
        goal = episode['image'][idx_goal]
        goal_state = episode['walker/world_zaxis'][idx_goal]

        if (goal.shape[0]//3)!=self.tile:
            goal = np.tile(goal,(self.tile,1,1))
        else:
            goal = goal[:self.tile*3,:,:]
 
        for i in range(self._nstep):
                step_reward = my_reward(episode["action"][idx+i], episode['walker/world_zaxis'][idx+i][:2] , goal_state[:2])*2
                # else:
                #     step_reward = -np.linalg.norm(episode["state"][idx+i][:self.index]-goal_state[:self.index])
                reward += discount * step_reward
                discount *= self._discount * self._discount

        return (obs, obs_state, action, reward, discount, next_obs, next_obs_state, goal, goal_state)


    def _sample(self):
        
        if self._load_every!=0:
            try:
                print('trying to load')
                self._load()
            except:
                traceback.print_exc()

            self._samples_since_last_load += 1
        
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode)) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]
        reward = episode["reward"][idx]
        discount = episode["discount"][idx] * self._discount
        reward = my_reward(action, next_obs, np.array((0.15, 0.15)))
        return (obs, action, reward, discount, next_obs)

    def _sample_gym(self):

        if self._load_every!=0:
            try:
                print('trying to load')
                self._load()
            except:
                traceback.print_exc()

            self._samples_since_last_load += 1

        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1

        # if self.prio_starts:
        # lower -= int(self.chunk * self.prio_starts)
        # if self.prio_ends:
        # upper += int(self.chunk * self.prio_ends)

        state = []
        next_state = []
        #if need to change obs_spec_keys that are used, then use original implemnetation 
        #from https://github.com/danijar/director/blob/a6a649efedc25a9d194155682bf39163fba4b5a2/embodied/agents/director/nets.py#L172

        obs = episode['image'][idx - 1]
        next_obs = episode['image'][idx + self._nstep - 1]
        for key in self._storage.obs_spec_keys():
            state_ = episode[key][idx - 1]
            next_state_ = episode[key][idx + self._nstep - 1]
            state.append(state_)
            next_state.append(next_state_)
        state = np.concatenate(state, axis=-1)
        next_state = np.concatenate(next_state, axis=-1)

        #joint states
        # state = {k: episode[k][idx-1] for k in self._storage.obs_spec_keys()}
        # chunk['is_first'] = np.zeros(len(chunk['action']), bool)
        # chunk['is_first'][0] = True

        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += self._discount * step_reward
            discount *= self._discount * self._discount

        return (obs, state, action, reward, next_obs, next_state, *meta)


                
    def __iter__(self):
        while True:
            if self.inv:
                if self.gym is False:
                    yield self._sample_inv()
                else:
                    yield self._sample_inv_gym()
            else:
                if self.gym is False:
                    yield self._sample()
                else:
                    yield self._sample_gym()

    def parse_dataset(self, start_ind=0, end_ind=-1,goal_state=None, proto_goal=False):
        states = []
        actions = []
        rewards = []
        episode_name = []
        index = []
        proto=[]
        if goal_state:
            goal_states=[]
        if len(self._episode_fns)>0:
            for eps_fn in tqdm.tqdm(self._episode_fns[start_ind:end_ind]):
                episode = self._episodes[eps_fn]
                ep_len = next(iter(episode.values())).shape[0] - 1
                for idx in range(ep_len):
                    
                    if self.pixels:
                        states.append(episode["state"][idx - 1][None])
                    else:
                        states.append(episode["observation"][idx - 1][None])
                    actions.append(episode["action"][idx][None])
                    rewards.append(episode["reward"][idx][None])
                    episode_name.append(str(eps_fn))
                    
                    if proto_goal:
                        proto.append(episode["observation"][idx - 1][None])
                    
                    index.append(np.array([idx]))
                    
                    if goal_state:
                        goal_states.append((episode["goal_state"][idx][None]))
            
            if goal_state:
      
                return (np.concatenate(states,0),
                        np.concatenate(actions, 0),
                        np.concatenate(rewards, 0),
                        np.concatenate(goal_states, 0),
                        episode_name,
                        index
                        )
            elif proto:
                return (np.concatenate(states,0),
                        np.concatenate(proto,0),
                        np.concatenate(actions, 0),
                        np.concatenate(rewards, 0),
                        episode_name,
                        index
                        )
            else:
                print('states', len(states))
                return (np.concatenate(states,0),
                        np.concatenate(actions, 0),
                        np.concatenate(rewards, 0),
                        episode_name,
                        index
                        )
        else:
            return ('', '')

def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(int(seed))

def make_replay_buffer(
    replay_dir,
    max_size,
    batch_size,
    num_workers,
    discount,
    offset=100,
    goal=False,
    relabel=False,
    replay_dir2=False,
    obs_type='state',
    offline=False,
    hybrid=False,
    hybrid_pct=0,
    nstep=1,
    eval=False,
    load_once=True,
    inv=False,
    goal_offset=1, 
    model_step=None,
    model_step_lb=None,
    pmm=True,
    reverse=True,
    gym=False,
    obs_spec_keys=None,
    tile=1):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = OfflineReplayBuffer(
        replay_dir,
        max_size_per_worker,
        num_workers,
        discount,
        offset,
        goal=goal,
        replay_dir2=replay_dir2,
        obs_type=obs_type,
        offline=offline,
        hybrid=hybrid,
        hybrid_pct=hybrid_pct,
        nstep=nstep,
        load_every=0,
        eval=eval,
        load_once=load_once,
        inv=inv,
        goal_offset=goal_offset,
        model_step=model_step,
        pmm=pmm,
        reverse=reverse,
        gym=gym,
        obs_spec_keys=obs_spec_keys,
        tile=tile
    )
    iterable._load()
    loader = torch.utils.data.DataLoader(
            iterable,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=_worker_init_fn)


    return loader

def make_replay_offline(
    replay_dir,
    max_size,
    batch_size,
    num_workers,
    discount,
    offset=100,
    goal=False,
    relabel=False,
    replay_dir2=False,
    obs_type='state',
    offline=False,
    hybrid=False,
    hybrid_pct=0,
    nstep=1,
    eval=False,
    load_once=True,
    inv=False,
    goal_offset=1, 
    model_step=None,
    model_step_lb=None,
    pmm=True,
    reverse=True,
    gym=False):
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = OfflineReplayBuffer(
        replay_dir,
        max_size_per_worker,
        num_workers,
        discount,
        offset,
        goal=goal,
        replay_dir2=replay_dir2,
        obs_type=obs_type,
        offline=offline,
        hybrid=hybrid,
        hybrid_pct=hybrid_pct,
        nstep=nstep,
        load_every=0,
        eval=eval,
        load_once=load_once,
        inv=inv,
        goal_offset=goal_offset,
        model_step=model_step,
        pmm=pmm,
        reverse=reverse,
        gym=gym
    )
    iterable._load()

    return iterable



def make_replay_loader(
    storage,  storage2, max_size, batch_size, num_workers, save_snapshot, nstep, discount, goal, hybrid=False, obs_type='state', hybrid_pct=0, actor1=False, replay_dir2=False,goal_proto=False, agent=None, neg_reward=False,return_iterable=False, sl=False, asym=False, loss=False, test=False, tile=1, pmm=True, obs_shape=4, general=False, inv=False,
goal_offset=1, gym=False):
    max_size_per_worker = max_size // max(1, num_workers)
    iterable = ReplayBuffer(
        storage,
        storage2,
        max_size_per_worker,
        num_workers,
        nstep,
        discount,
        goal=goal,
        hybrid=hybrid,
        obs_type = obs_type,
        hybrid_pct=hybrid_pct,
        actor1 = actor1,
        replay_dir2=replay_dir2,
        goal_proto=goal_proto,
        agent=agent,
        neg_reward=neg_reward,
        sl=sl,
        asym=asym,
        loss=loss,
        test=test,
        tile=tile,
        fetch_every=1000,
        save_snapshot=save_snapshot,
        pmm=pmm,
        obs_shape=obs_shape,
        general=general,
        inv=inv,
        goal_offset=goal_offset,
        gym=gym)

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    if return_iterable:
        return iterable, loader
    else:
        return loader

def ndim_grid(ndims, space):
    L = [np.linspace(-.25,.25,space) for i in range(ndims)]
    return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T

def collate_tensor_fn(batch, *, collate_fn_map):
    return torch.stack(batch, 0)

def custom_collate(batch):
    collate_map = {torch.Tensor: collate_tensor_fn}
    return torch.utils.data._utils.collate.collate(batch, collate_fn_map=collate_map)

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
        last = str(replay_dir).split('/')[-1]
        self._replay_dir2 = replay_dir.parent / last / "buffer_copy"
        replay_dir.mkdir(exist_ok=True)
        (replay_dir.parent / last / "buffer_copy").mkdir(exist_ok=True)
        self._current_episode = defaultdict(list)
        self._current_episode_goal = defaultdict(list)
        self._preload()
        self.state_visitation_proto = np.zeros((60,60))
        self.state_visitation_proto_pct = np.zeros((20,20))
        self.state_visitation_gc = np.zeros((60,60))
        self.state_visitation_gc_pct = np.zeros((20,20))
        self.reward_matrix = np.zeros((60,60))
        self.goal_state_matrix = np.zeros((60,60))
    def __len__(self):
        return self._num_transitions

    def add(self, time_step, state=None, meta=None,pixels=False, last=False, pmm=True):
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
                    idx_x = int(tmp_state[0])+29
                    idx_y = int(tmp_state[1])+29
                    self.state_visitation_proto[idx_x,idx_y]+=1
                
                    tmp_state = tmp_state/3
                    idx_x = int(tmp_state[0])+9
                    idx_y = int(tmp_state[1])+9
                    self.state_visitation_proto_pct[idx_x,idx_y]+=1
                
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
            self._store_episode(episode, actor1=False)
            print('storing episode, no goal')

    def add_goal(self, time_step, meta, goal, time_step_no_goal=False,goal_state=False,pixels=False, last=False, asym=False):
        for key, value in meta.items():
            self._current_episode_goal[key].append(value)
        for spec in self._data_specs:
            if spec.name == 'observation' and pixels:
                value = time_step_no_goal[spec.name]
                self._current_episode_goal['observation'].append(value['pixels'])
                self._current_episode_goal['state'].append(value['observations'])
                
                
                tmp_state = value['observations']*100
                idx_x = int(tmp_state[0])+29
                idx_y = int(tmp_state[1])+29
                self.state_visitation_gc[idx_x,idx_y]+=1
                
                tmp_state = tmp_state/3
                idx_x = int(tmp_state[0])+9
                idx_y = int(tmp_state[1])+9
                self.state_visitation_gc_pct[idx_x,idx_y]+=1
                
            else:
                value = time_step[spec.name]
                if np.isscalar(value):
                    value = np.full(spec.shape, value, spec.dtype)
                assert spec.shape == value.shape and spec.dtype == value.dtype
                self._current_episode_goal[spec.name].append(value)
                
                if spec.name == 'reward' and pixels:
                    value = time_step['observation']
                    tmp_state = value['observations']*100
                    idx_x = int(tmp_state[0])+29
                    idx_y = int(tmp_state[1])+29
                    self.reward_matrix[idx_x,idx_y]+=time_step['reward']
                
        if pixels and asym==False:
            goal = np.transpose(goal, (2,0,1))
            self._current_episode_goal['goal_state'].append(goal_state)
            idx_x = int(goal_state[0]*100)+29
            idx_y = int(goal_state[1]*100)+29
            self.goal_state_matrix[idx_x,idx_y]+=1
            
        elif pixels and asym:
            self._current_episode_goal['goal_state'].append(goal_state)
            idx_x = int(goal_state[0]*100)+29
            idx_y = int(goal_state[1]*100)+29
            self.goal_state_matrix[idx_x,idx_y]+=1
        self._current_episode_goal['goal'].append(goal)

        if time_step.last() or last:
            print('replay last')
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
            print('storing episode, w/ goal')
            
    def add_goal_general(self, time_step, state, meta, goal, goal_state, time_step_no_goal, pixels=False, last=False, asym=False):
        #assert goal.shape[0]==9 and goal.shape[1]==84 and goal.shape[2]==84
        if time_step_no_goal is not None:
            pmm=True
        else:
            pmm=False

        for key, value in meta.items():
            self._current_episode_goal[key].append(value)
            
        for spec in self._data_specs:
            if spec.name == 'observation' and pixels:
                if pmm:
                    value = time_step_no_goal[spec.name]
                    self._current_episode_goal['observation'].append(value['pixels'])
                    self._current_episode_goal['state'].append(value['observations'])


                    tmp_state = value['observations']*100
                    idx_x = int(tmp_state[0])+29
                    idx_y = int(tmp_state[1])+29
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
                    idx_x = int(tmp_state[0])+29
                    idx_y = int(tmp_state[1])+29
                    self.reward_matrix[idx_x,idx_y]+=time_step['reward']
                
        if pixels and asym==False and pmm:
            
            #goal = np.transpose(goal, (2,0,1))
            self._current_episode_goal['goal_state'].append(goal_state)
            idx_x = int(goal_state[0]*100)+29
            idx_y = int(goal_state[1]*100)+29
            self.goal_state_matrix[idx_x,idx_y]+=1
            
        elif pixels and asym==False:
            self._current_episode_goal['goal_state'].append(goal_state)
            
        self._current_episode_goal['goal'].append(goal)

        if time_step.last() or last:
            
            print('replay last')
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
            if np.any(episode['goal']<0):
                import IPython as ipy; ipy.embed(colors='neutral')
            if pixels:
                value = self._current_episode_goal['goal_state']
                episode['goal_state'] = np.array(value, np.float64)
                
            self._current_episode_goal = defaultdict(list)
            self._store_episode(episode, actor1=True)
            print('storing episode, w/ goal, general')
            
            
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
                    idx_x = int(tmp_state[0])+29
                    idx_y = int(tmp_state[1])+29
                    self.state_visitation_gc[idx_x,idx_y]+=1
                    tmp_state = tmp_state/3
                    idx_x = int(tmp_state[0])+9
                    idx_y = int(tmp_state[1])+9
                    self.state_visitation_gc_pct[idx_x,idx_y]+=1

            elif spec.name=='reward' and pmm:
                
                value = np.array([reward]).reshape((-1,))*2
                self._current_episode_goal['reward'].append(value)

                value = time_step['observation']
                tmp_state = value['observations']*100
                idx_x = int(tmp_state[0])+29
                idx_y = int(tmp_state[1])+29
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
            idx_x = int(goal_state[0]*100)+29
            idx_y = int(goal_state[1]*100)+29
            self.goal_state_matrix[idx_x,idx_y]+=1 

        if time_step.last() or last:
            print('replay last')
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
            print('storing episode, w/ goal, proto')
         

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
        model_step=False,
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
        general=False):
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
        if model_step:
            self.model_step = int(int(model_step)/500)


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
                #print('eps2', eps_fn2)
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
            
            if self.test:
                #if idx < episode_len(episode)//2:
                #    rand_idx = np.random.randint(episode_len(episode)-50, episode_len(episode))
                #    rand_obs = episode["observation"][rand_idx]
                #    rand_obs_state = episode["state"][rand_idx]
                #else:
                #    rand_idx = np.random.randint(0, 50)
                #    rand_obs = episode["observation"][rand_idx]
                #    rand_obs_state = episode["state"][rand_idx]
                next_obs_state = episode["state"][idx + self._nstep - 1]    
                obs_state =  episode["state"][idx - 1]
                
                episode = self._sample_episode()
                idx = np.random.randint(0, episode_len(episode))
                rand_obs = episode['observation'][idx - 1]
                rand_obs_state = episode["state"][idx-1]
                return (obs, obs_state, action, reward, discount, next_obs, next_obs_state, rand_obs, rand_obs_state, *meta)
            else:
                obs_state =  episode["state"][idx - 1]
                next_obs_state = episode["state"][idx + self._nstep - 1]
                
                episode = self._sample_episode()
                idx = np.random.randint(0, episode_len(episode))
                rand_obs = episode['observation'][idx - 1]
                return (obs, obs_state, action, reward, discount, next_obs, next_obs_state, rand_obs, *meta)	

        elif self.test:
            obs_state =  episode["state"][idx - 1]
            return (obs, obs_state, action, reward, discount, next_obs, *meta)
        else:
            return (obs, action, reward, discount, next_obs, next_obs_state, *meta)
    

    def _sample_sl(self):
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
        obs_state = episode["state"][idx - 1]

        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount

        if self.goal:
            goal = episode["goal"][idx-1]
            goal_state=episode["goal_state"][idx-1]
            if self.pixels and self.goal_proto==False:
                goal = np.tile(goal,(self.tile,1,1))
            return (obs, obs_state, action, reward, discount, next_obs, goal, goal_state, *meta)
        else:
            return (obs, obs_state, action, reward, discount, next_obs, *meta) 
        
        
    def _sample_asym(self):
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
        next_state = episode["state"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        obs_state = episode["state"][idx - 1]

        for i in range(self._nstep):
            step_reward = episode["reward"][idx + i]
            reward += discount * step_reward
            discount *= episode["discount"][idx + i] * self._discount
            
        goal_state=episode["goal"][idx-1]
        
        #goal = goal[None,:,:]
        return (obs, obs_state, action, reward, discount, next_obs, next_state, goal_state, *meta)
        
    def _sample_goal_hybrid(self):
        #print(self._episode_fns)
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
        obs_state = episode["state"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]
        next_obs_state = episode["state"][idx + self._nstep - 1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        offset = 0 
        #hybrid where we use hybrid_pct of relabeled gc data in each batch
        #100-hybrid_pct*10 is just original gc data
        key = np.random.uniform()


        if key > self.hybrid_pct:
            #make sure we're using an episode collected by gc 
            while 'goal' not in episode.keys():
                episode = self._sample_episode()
            idx = np.random.randint(0, episode_len(episode)-self._nstep+1) + 1 
            obs = episode["observation"][idx - 1]
            obs_state = episode["state"][idx - 1]
            action = episode["action"][idx]
            next_obs = episode["observation"][idx + self._nstep - 1]
            next_obs_state = episode["state"][idx + self._nstep - 1]
            reward = np.zeros_like(episode["reward"][idx])
            discount = np.ones_like(episode["discount"][idx])
            offset = 0 
            goal = episode["goal"][idx-1]
            goal_state = episode["goal_state"][idx-1, :2]
            
            if self.asym:
                goal_state = episode["goal"][idx-1]
            
            if self.pixels and self.goal_proto==False and self.asym==False and self.general==False:
                goal = np.tile(goal,(self.tile,1,1))
            
            for i in range(self._nstep):
                step_reward = episode["reward"][idx + i]
                reward += discount * step_reward
                discount *= episode["discount"][idx + i] * self._discount
            
            
        elif key <= self.hybrid_pct and self.goal_proto==False:
            idx = np.random.randint(episode_len(episode)-self._nstep) + 1
            obs = episode["observation"][idx-1]
            obs_state = episode["state"][idx-1]
            action = episode["action"][idx]
            next_obs = episode['observation'][idx + self._nstep - 1]
            next_obs_state = episode['state'][idx + self._nstep - 1]
            idx_goal = np.random.randint(idx + self._nstep - 1,episode_len(episode))
            
            offset = idx_goal - idx
            goal = episode["observation"][idx_goal]
            goal_state = episode["state"][idx_goal,:2]
            for i in range(self._nstep):
                if self.pmm:
                    step_reward = my_reward(episode["action"][idx+i],episode["state"][idx+i] , goal_state[:2])*2
                else:
                    step_reward = -np.linalg.norm(episode["state"][idx+i][:self.index]-goal_state[:self.index])
                reward += discount * step_reward
                #reward += discount * discount * step_reward
                discount *= episode["discount"][idx+i] * self._discount
        elif key <= self.hybrid_pct and self.goal_proto:
            #import IPython as ipy; ipy.embed(colors='neutral')
            idx = np.random.randint(episode_len(episode)-self._nstep)+1
            obs = episode["observation"][idx-1]
            obs_state = episode["state"][idx-1]
            action = episode["action"][idx]
            next_obs = episode['observation'][idx + self._nstep - 1]
            next_obs_state = episode['state'][idx + self._nstep - 1]
            idx_goal = np.random.randint(idx + self._nstep - 1,episode_len(episode))
            goal=episode['observation'][idx_goal]
            z = episode["observation"][idx_goal][None,:]
            protos = self.agent.protos.weight.data.detach().clone().cpu().numpy()
            z_to_proto = np.linalg.norm(z[:, None, :] - protos[None, :, :], axis=2, ord=2)
            _ = np.argsort(z_to_proto, axis=1)[:,0]
            goal = protos[_].reshape((protos.shape[1],))
            offset=idx_goal-idx
            for i in range(self._nstep):
                obs_to_proto = np.linalg.norm(z[:, None, :] - protos[None, :, :], axis=2, ord=2)
                dists_idx = np.argsort(obs_to_proto, axis=1)[:,0]
                
                if np.array_equal(goal,protos[dists_idx]):
                    reward=1
                else:
                    reward=0
                    
                step_reward = reward*2
                reward += discount * step_reward
                discount *= episode["discount"][idx+i] * self._discount  
        else:
            print('sth went wrong in replay buffer')
        
        if self.loss:
            episode = self._sample_episode()
            idx = np.random.randint(0, episode_len(episode))
            rand_obs = episode['observation'][idx - 1]
        
        goal = goal.astype(int)
        reward = np.array(reward).astype(float)
        offset = np.array(offset).astype(float)
        if self.sl:
            return (obs, obs_state, action, reward, discount, next_obs, goal, goal_state)
        elif self.loss:
            return (obs, obs_state, action, reward, discount, next_obs, next_obs_state, goal, rand_obs, *meta)
        elif self.asym:
            
            return (obs, obs_state, action, reward, discount, next_obs, next_obs_state, goal_state, *meta)
        else:
            return (obs, action, reward, discount, next_obs, goal, *meta)
    
    def _sample_goal_offline(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode)) + 1
        meta = []
        for spec in self._storage._meta_specs:
            meta.append(episode[spec.name][idx - 1])

        obs = episode["observation"][idx - 1]

        action = episode["action"][idx]
        next_obs = episode["observation"][idx]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])
        
        iz = np.random.randint(idx, episode_len(episode))+1
        goal = episode["observation"][iz]
        goal_state = episode["state"][iz]
        reward = my_reward(action,episode["state"][idx] , goal_state)*2
        #for i in range(self._nstep):
        #    step_reward = my_reward(action,episode["state"][idx+i] , goal_state)
        #    reward += discount * step_reward
        #    discount *= episode["discount"][idx+i] * self._discount

        return (obs, action, reward, discount, next_obs, goal, *meta)


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

    def __iter__(self):
        while True:
            if self.sl and self.hybrid==False:
                yield self._sample_sl()
            elif self.asym and self.hybrid==False:
                yield self._sample_asym()
            elif self.hybrid:
                yield self._sample_goal_hybrid()
            else:
                yield self._sample()


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
        model_step=False,
        model_step_lb = False,
        replay_dir2=False,
        obs_type='state',
        hybrid=False,
        hybrid_pct=0,
        offline=False,
        nstep=1,
        load_every=1000,
        eval=False,
        load_once=False):

        self._env = env
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
        self.vae = vae
        self.goal_array = []
        self._goal_array = False
        self.obs = []
        self.model_step = int(int(model_step)/500)
        self.model_step_lb = int(int(model_step_lb)/500)
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
        print('self._load_every', self._load_every)
        self._samples_since_last_load = load_every
        self.load_once=load_once
        self.switch=False

        if obs_type == 'pixels':
            self.pixels = True      
        else:
            self.pixels = False
        self.eval = eval

    def _load(self, relabel=False):
        print("Labeling data...")
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob("*.npz"), reverse=True)
    
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
            print(eps_fn)
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
    
    def _sample_her(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode)) +1
        action = episode["action"][idx]
        obs = episode["observation"][idx - 1]
        next_obs = episode["observation"][idx]
        reward = episode["reward"][idx]
        discount = episode["discount"][idx] * self._discount
        goal = episode["goal"][idx]
        future_offset=0
        if np.random.uniform()<self.hybrid_pct:
            future_offset = np.random.uniform()*(episode_len(episode) - idx +1)
            future_offset = future_offset.astype(int)
            goal = episode["goal"][idx]
            
            if self.pixels:
            
                reward = my_reward(action, episode["state"][idx], episode["goal_state"][idx])
            else:
                reward = my_reward(action, next_obs, goal)
            
        return (obs, action, reward, discount, next_obs, goal, future_offset)
            
    
    def _sample(self):
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode)) + 1
        obs = episode["observation"][idx - 1]
        action = episode["action"][idx]
        next_obs = episode["observation"][idx + self._nstep - 1]
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
        next_obs = episode["observation"][idx + self._nstep - 1]
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
        

    def _sample_pixel_goal(self, time_step):
        episode = self._sample_episode()
        if time_step<1000000:
            idx = np.random.randint(int(time_step/10000)*5, min((int(time_step/10000)+1)*5, 500))
        else:
            idx = np.random.randint(250,500)
        obs = episode["observation"][idx]
        state = episode["state"][idx][:2]
        return obs, state

    
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

    def _sample_goal_hybrid(self):
        try:
            self._load()
        except:
            traceback.print_exc()
        self._samples_since_last_load += 1

        episode = self._sample_episode()
        idx = np.random.randint(0, episode_len(episode) - self._nstep+1) + 1
        obs = episode["observation"][idx - 1]

        action = episode["action"][idx]
        next_obs = episode["observation"][idx+self._nstep-1]
        reward = np.zeros_like(episode["reward"][idx])
        discount = np.ones_like(episode["discount"][idx])

        if 'goal' in episode.keys():
            goal = episode["goal"][idx]
            if self.goal_proto==False:
                goal = np.tile(goal,(self.tile,1,1))
            for i in range(self._nstep):
                step_reward = episode["reward"][idx + i]
                reward += discount * step_reward
                discount *= episode["discount"][idx + i] * self._discount

        else:
       
            self.count+=1
            #??
            if self.count==1000*self.hybrid_pct:
                self.iz +=1
            if self.iz>500-self._nstep:
                self.iz=1

            obs = episode["observation"][self.iz-1]
            action = episode["action"][self.iz]
            next_obs = episode['observation'][self.iz+self._nstep-1]
            idx_goal = np.random.randint(self.iz,min(self.iz+50, 499))
            goal = episode["observation"][idx_goal][None, :]
            print('goal', goal.shape)

            for i in range(self._nstep):
                for z in range(2):
                    step_reward = my_reward(episode["action"][self.iz+i],episode["state"][self.iz+i] , goal_state[:2])
                    reward += discount * step_reward
                    discount *= episode["discount"][idx+i] * self._discount 
        
        reward = np.array(reward).astype(float)
        goal = goal.astype(int)
        obs = np.array(obs)
        return (obs, action, reward, discount, next_obs, goal)
    
       
                
    def __iter__(self):
        while True:
            if (self.offline and self.goal) or (self.hybrid and self.goal):
                yield self._sample_her()
            else:
                yield self._sample()


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
    env,
    replay_dir,
    max_size,
    batch_size,
    num_workers,
    discount,
    offset=100,
    goal=False,
    vae=False,
    relabel=False,
    model_step=False,
    model_step_lb=False,
    replay_dir2=False,
    obs_type='state',
    offline=False,
    hybrid=False,
    hybrid_pct=0,
    nstep=1,
    eval=False,
    load_every=1000):
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
        model_step=model_step,
        model_step_lb=model_step_lb,
        replay_dir2=replay_dir2,
        obs_type=obs_type,
	offline=offline,
        hybrid=hybrid,
        hybrid_pct=hybrid_pct,
        nstep=nstep,
        load_every=load_every,
        eval=eval
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

def make_replay_offline(
    env,
    replay_dir,
    max_size,
    batch_size,
    num_workers,
    discount,
    offset=100,
    goal=False,
    vae=False,
    relabel=False,
    model_step=False,
    model_step_lb=False,
    replay_dir2=False,
    obs_type='state',
    offline=False,
    hybrid=False,
    hybrid_pct=0,
    nstep=1,
    eval=False,
    load_once=True):
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
        model_step=model_step,
        model_step_lb=model_step_lb,
        replay_dir2=replay_dir2,
        obs_type=obs_type,
        offline=offline,
        hybrid=hybrid,
        hybrid_pct=hybrid_pct,
        nstep=nstep,
        load_every=0,
        eval=eval,
        load_once=load_once
    )
    iterable._load()
	
    return iterable



def make_replay_loader(
    storage,  storage2, max_size, batch_size, num_workers, save_snapshot, nstep, discount, goal, hybrid=False, obs_type='state', hybrid_pct=0, actor1=False, replay_dir2=False,model_step=False,goal_proto=False, agent=None, neg_reward=False,return_iterable=False, sl=False, asym=False, loss=False, test=False, tile=1, pmm=True, obs_shape=4, general=False):
    print('h1', hybrid)
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
        model_step=model_step,
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
        general=general
        )

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


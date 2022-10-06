import numpy as np

#https://github.com/rail-berkeley/rlkit/blob/354f14c707cc4eb7ed876215dd6235c6b30a2e2b/rlkit/samplers/data_collector/path_collector.py#L99

# class MdpPathCollector(PathCollector):
#     def __init__(
#             self,
#             env,
#             policy,
#             max_num_epoch_paths_saved=None,
#             render=False,
#             render_kwargs=None,
#             rollout_fn=rollout,
#             save_env_in_snapshot=True,
#     ):
#         if render_kwargs is None:
#             render_kwargs = {}
#         self._env = env
#         self._policy = policy
#         self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
#         self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
#         self._render = render
#         self._render_kwargs = render_kwargs
#         self._rollout_fn = rollout_fn

#         self._num_steps_total = 0
#         self._num_paths_total = 0

#         self._save_env_in_snapshot = save_env_in_snapshot

#     def collect_new_paths(
#             self,
#             max_path_length,
#             num_steps,
#             discard_incomplete_paths,
#     ):
#         paths = []
#         num_steps_collected = 0
#         while num_steps_collected < num_steps:
#             max_path_length_this_loop = min(  # Do not go over num_steps
#                 max_path_length,
#                 num_steps - num_steps_collected,
#             )
#             path = self._rollout_fn(
#                 self._env,
#                 self._policy,
#                 max_path_length=max_path_length_this_loop,
#                 render=self._render,
#                 render_kwargs=self._render_kwargs,
#             )
#             path_len = len(path['actions'])
#             if (
#                     path_len != max_path_length
#                     and not path['terminals'][-1]
#                     and discard_incomplete_paths
#             ):
#                 break
#             num_steps_collected += path_len
#             paths.append(path)
#         self._num_paths_total += len(paths)
#         self._num_steps_total += num_steps_collected
#         self._epoch_paths.extend(paths)
#         return paths

#     def get_epoch_paths(self):
#         return self._epoch_paths

#     def end_epoch(self, epoch):
#         self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

#     def get_diagnostics(self):
#         path_lens = [len(path['actions']) for path in self._epoch_paths]
#         stats = OrderedDict([
#             ('num steps total', self._num_steps_total),
#             ('num paths total', self._num_paths_total),
#         ])
#         stats.update(create_stats_ordered_dict(
#             "path length",
#             path_lens,
#             always_show_all_stats=True,
#         ))
#         return stats

#     def get_snapshot(self):
#         snapshot_dict = dict(
#             policy=self._policy,
#         )
#         if self._save_env_in_snapshot:
#             snapshot_dict['env'] = self._env
#         return snapshot_dict


# class GoalConditionedPathCollector(MdpPathCollector):
#     def __init__(
#             self,
#             *args,
#             observation_key='observation',
#             desired_goal_key='desired_goal',
#             goal_sampling_mode=None,
#             **kwargs
#     ):
#         def obs_processor(o):
#             return np.hstack((o[observation_key], o[desired_goal_key]))

#         rollout_fn = partial(
#             rollout,
#             preprocess_obs_for_policy_fn=obs_processor,
#         )
#         super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
#         self._observation_key = observation_key
#         self._desired_goal_key = desired_goal_key
#         self._goal_sampling_mode = goal_sampling_mode

#     def collect_new_paths(self, *args, **kwargs):
#         self._env.goal_sampling_mode = self._goal_sampling_mode
#         return super().collect_new_paths(*args, **kwargs)

#     def get_snapshot(self):
#         snapshot = super().get_snapshot()
#         snapshot.update(
#             observation_key=self._observation_key,
#             desired_goal_key=self._desired_goal_key,
#         )
#         return snapshot


# class VAEWrappedEnvPathCollector(GoalConditionedPathCollector):
#     def __init__(
#             self,
#             env,
#             policy,
#             decode_goals=False,
#             **kwargs
#     ):
#         """Expects env is VAEWrappedEnv"""
#         super().__init__(env, policy, **kwargs)
#         self._decode_goals = decode_goals

#     def collect_new_paths(self, *args, **kwargs):
#         self._env.decode_goals = self._decode_goals
#         return super().collect_new_paths(*args, **kwargs)





class PathBuilder(dict):
    """
    Usage:
    ```
    path_builder = PathBuilder()
    path.add_sample(
        observations=1,
        actions=2,
        next_observations=3,
        ...
    )
    path.add_sample(
        observations=4,
        actions=5,
        next_observations=6,
        ...
    )
    path = path_builder.get_all_stacked()
    path['observations']
    # output: [1, 4]
    path['actions']
    # output: [2, 5]
    ```
    Note that the key should be "actions" and not "action" since the
    resulting dictionary will have those keys.
    """

    def __init__(self):
        super().__init__()
        self._path_length = 0

    def add_all(self, **key_to_value):
        for k, v in key_to_value.items():
            if k not in self:
                self[k] = [v]
            else:
                self[k].append(v)
        self._path_length += 1

    def get_all_stacked(self):
        output_dict = dict()
        for k, v in self.items():
            output_dict[k] = stack_list(v)
        return output_dict

    def __len__(self):
        return self._path_length


def stack_list(lst):
    if isinstance(lst[0], dict):
        return lst
    else:
        return np.array(lst)
    
    
    

class GoalConditionedStepCollector():
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            observation_key='observation',
            desired_goal_key='desired_goal',
    ):

        self._env = env
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key

        self._num_steps_total = 0
        self._num_paths_total = 0
        self._obs = None  # cache variable
        
    def start_collection(self):
        return None
    
    def end_collection(self):
        return None
    
    def collect_new_steps(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        for _ in range(num_steps):
            self.collect_one_step(max_path_length, discard_incomplete_paths)
    
    
    def collect_one_step(
            self,
            max_path_length,
            discard_incomplete_paths,
    ):
        if self._obs is None:
            self._start_new_rollout()

        new_obs = np.hstack((
            self._obs[self._observation_key],
            self._obs[self._desired_goal_key],
        ))
        
        action, agent_info = self._policy.get_action(new_obs)
        next_ob, reward, goal = (
            self._env.step(action)
        )

        reward = np.array([reward])
        # store path obs
        self._current_path_builder.add_all(
            observations=self._obs,
            actions=action,
            rewards=reward,
            next_observations=next_ob,
            terminals=terminal,
            agent_infos=agent_info,
            env_infos=env_info,
        )
        if terminal or len(self._current_path_builder) >= max_path_length:
            self._handle_rollout_ending(max_path_length,
                                        discard_incomplete_paths)
            self._start_new_rollout()
        else:
            self._obs = next_ob

            
    def _start_new_rollout(self):
        return None


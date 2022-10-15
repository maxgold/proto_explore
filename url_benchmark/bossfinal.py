import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs
import matplotlib.pyplot as plt
from kdtree import KNN

import dmc
import torch.nn.functional as F
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from offline_replay_buffer import make_replay_buffer
from video import TrainVideoRecorder, VideoRecorder
from agent.expert import ExpertAgent

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS

from agent.ddpg import RefinedDDPGAgent


def get_state_embeddings(agent, states):
    with torch.no_grad():
        s = agent.encoder(states)
        s = agent.predictor(s)
        s = agent.projector(s)
        s = F.normalize(s, dim=1, p=2)
    return s


def proto2states(agent, states=None):
    grid = np.meshgrid(np.arange(-0.3, 0.3, 0.01), np.arange(-0.3, 0.3, 0.01))
    grid = np.concatenate((grid[0][:, :, None], grid[1][:, :, None]), -1)
    grid = grid.reshape(-1, 2)
    grid = np.c_[grid, np.zeros_like(grid)]
    grid = torch.tensor(grid).cuda().float()
    grid_embeddings = get_state_embeddings(agent, grid)
    protos = agent.protos.weight.data.clone()
    protos = F.normalize(protos, dim=1, p=2)
    dist_mat = torch.cdist(protos, grid_embeddings)
    closest_points = dist_mat.argmin(-1)
    return grid[closest_points, :2].cpu()


### STEPS
# 1. load agent
# 2. load pretrained goal agent
# 3. Set up environment
# 4. Map prototypes to their closest point in state space via a grid
# 5. Relabel all these states with reward and choose highest one
# 6. Run goal conditioned RL to reach that prototype




def make_refined_agent(goal_policy, obs_type, obs_spec, action_spec, num_expl_steps, cfg):

    agent = RefinedDDPGAgent(
        goal_policy,
        cfg.weight,
        "rddpg",
        False,
        obs_type,
        obs_spec.shape,
        action_spec.shape,
        cfg.device,
        cfg.agent.lr,
        cfg.agent.feature_dim,
        cfg.agent.hidden_dim,
        cfg.agent.critic_target_tau,
        num_expl_steps,
        cfg.agent.update_every_steps,
        cfg.agent.stddev_schedule,
        cfg.agent.nstep,
        cfg.agent.batch_size,
        cfg.agent.stddev_clip,
        cfg.agent.init_critic,
        cfg.agent.use_tb,
        False,
    )
    return agent


class GoalHelper:
    def __init__(self, goal_agent, goal):
        self.goal_agent = goal_agent
        self.goal = goal

    def __call__(self, obs):
        if (obs.shape[0] > 1) or (len(obs.shape) > 2):
            goal = self.goal.tile((obs.shape[0],1))
        else:
            obs = obs[0]
            goal = self.goal
        return self.goal_agent.act(obs, goal, {}, 0, eval_mode=True)



class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        #import IPython as ipy; ipy.embed(colors='neutral')
        d = torch.load(
            "/home/maxgold/workspace/explore/proto_explore/url_benchmark/models/states/point_mass_maze_reach_bottom_right/proto/1/snapshot_2000000.pt"
        )
        proto_agent = d["agent"]
        goal_agent = d["goal_agent"]
        #path_to_agent = "/home/maxgold/workspace/explore/proto_explore/output/2022.08.24/174419_gcac_gcsl_nohorizon/agent"
        #goal_agent = torch.load(path_to_agent)

        goal = (-.23, -.23)

        cand_states = proto2states(proto_agent)
        env = dmc.make("point_mass_maze_reach_custom_goal", seed=0, goal=goal)
        rewards = []
        for state in cand_states:
            with env.physics.reset_context():
                env.physics.set_state(np.r_[state, np.zeros(2)])
            reward = env.task.get_reward(env.physics)
            rewards.append(reward)

        istate = np.argmax(rewards)

        goal_state = cand_states[istate]

        goal_policy = GoalHelper(goal_agent, goal_state)

        self.agent = make_refined_agent(
            goal_policy,
            cfg.obs_type,
            env.observation_spec(),
            env.action_spec(),
            cfg.num_seed_frames // cfg.action_repeat,
            cfg
        )

        # create logger
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs

        self.train_env = dmc.make("point_mass_maze_reach_custom_goal", seed=cfg.seed, goal=goal)
        self.eval_env = dmc.make("point_mass_maze_reach_custom_goal", seed=cfg.seed, goal=goal)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)

                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)

            if hasattr(self.agent, "regress_meta"):
                repeat = self.cfg.action_repeat
                every = self.agent.update_task_every_step // repeat
                init_step = self.agent.num_init_steps
                if self.global_step > (
                        init_step // repeat) and self.global_step % every == 0:
                    meta = self.agent.regress_meta(self.replay_iter,
                                                   self.global_step)

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        domain, _ = self.cfg.task.split('_', 1)
        snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name

        def try_load(seed):
            snapshot = snapshot_dir / str(
                seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            if not snapshot.exists():
                return None
            with snapshot.open('rb') as f:
                payload = torch.load(f)
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        # otherwise try random seed
        while True:
            seed = np.random.randint(1, 11)
            payload = try_load(seed)
            if payload is not None:
                return payload
        return None

@hydra.main(config_path='.', config_name='finalboss')
def main(cfg):
    from bossfinal import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()



if __name__=="__main__":
    main()












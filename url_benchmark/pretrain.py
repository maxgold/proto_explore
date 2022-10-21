import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['HYDRA_FULL_ERROR']='1'
import seaborn as sns; sns.set_theme()
from pathlib import Path
import torch.nn.functional as F
import torch
import torch.nn as nn
import hydra
import numpy as np
import torch
import wandb
from dm_env import specs
import dmc
import utils
from scipy.spatial.distance import cdist
from logger import Logger, save
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_replay_buffer, ndim_grid, make_replay_offline
import matplotlib.pyplot as plt
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS

def make_agent(obs_type, obs_spec, action_spec, goal_shape, num_expl_steps, cfg, lr=.0001, hidden_dim=1024, batch_size=256, num_protos=512, update_gc=2, gc_only=False, offline=False, feature_dim=50, pred_dim=128):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    cfg.goal_shape = goal_shape
    cfg.lr = lr
    cfg.hidden_dim = hidden_dim
    cfg.batch_size = batch_size
    cfg.num_protos=num_protos
    cfg.update_gc=update_gc
    cfg.gc_only=gc_only
    cfg.offline=offline
    cfg.feature_dim = feature_dim
    cfg.pred_dim = pred_dim
    return hydra.utils.instantiate(cfg)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        work_path = str(os.getcwd().split('/')[-2])+'/'+str(os.getcwd().split('/')[-1])

        # create logger
        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type,
                str(cfg.seed), str(cfg.tmux_session),work_path 
            ])
            wandb.init(project="urlb", group=cfg.agent.name, name=exp_name)
 
        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        try:
            task = PRIMAL_TASKS[self.cfg.domain]
        except:
            task = self.cfg.domain
        self.train_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)
        self.eval_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, cfg.seed)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                (3,84,84),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent,
                                cfg.lr,
                                cfg.hidden_dim,
                                cfg.batch_size,
                                cfg.num_protos,
                                cfg.update_gc,
                                False,
                                cfg.offline,
                                cfg.feature_dim,
                                cfg.pred_dim)

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
                                                False,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount,
                                                goal=cfg.goal,
                                                obs_type=cfg.obs_type)
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb)

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
        heatmap = self.replay_storage.state_visitation_proto

        plt.clf()
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        ax.set_title(self.global_step)

        plt.savefig(f"./{self.global_step}_proto_heatmap.png")
        wandb.save(f"./{self.global_step}_proto_heatmap.png")


        heatmap_pct = self.replay_storage.state_visitation_proto_pct

        plt.clf()
        fig, ax = plt.subplots(figsize=(10,10))
        labels = np.round(heatmap_pct.T/heatmap_pct.sum()*100, 1)
        sns.heatmap(np.log(1 + heatmap_pct.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        ax.set_title(self.global_step)

        plt.savefig(f"./{self.global_step}_proto_heatmap_pct.png")
        wandb.save(f"./{self.global_step}_proto_heatmap_pct.png")
 
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            goal = np.random.sample((2,)) * .5 - .25
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    if self.cfg.goal:
                        if self.cfg.obs_type=='pixels':
                            action = self.agent.act(time_step.observation['pixels'],
                                            goal,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                        else:
                            action = self.agent.act(time_step.observation,
                                            goal,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                    else:
                        if self.cfg.obs_type=='pixels':
                            if self.cfg.use_predictor:
                                action = self.agent.act2(time_step.observation['pixels'],
                                            meta,
                                            self.global_step,
                                            eval_mode=True,
                                            proto=self.agent)
                            else:
                                action = self.agent.act2(time_step.observation['pixels'],
                                            meta,
                                            self.global_step,
                                            eval_mode=True) 
                        else:    
                            action = self.agent.act2(time_step.observation,
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
        if self.cfg.obs_type=='pixels':
            self.replay_storage.add(time_step, meta, True)
        else:
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
                if self.cfg.const_init==False:
                    task = PRIMAL_TASKS[self.cfg.domain]
                    rand_init = np.random.uniform(.02,.29,size=(2,))
                    sign = np.array([[1,1],[-1,1],[1,-1],[-1,-1]])
                    rand = np.random.randint(4)
                    self.train_env = dmc.make(task, self.cfg.obs_type, self.cfg.frame_stack,
                                                              self.cfg.action_repeat, self.cfg.seed, init_state=(rand_init[0]*sign[rand][0], rand_init[1]*sign[rand][1]))
                    print('sampled init', (rand_init[0]*sign[rand][0], rand_init[1]*sign[rand][1]))   
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                if self.cfg.obs_type=='pixels':

                    self.replay_storage.add(time_step, meta, True)
                else: 
                    self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()
            
            goal = np.random.sample((2,)) * .5 - .25
            meta = self.agent.update_meta(meta, self.global_step, time_step)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                if self.cfg.goal:
                    if self.cfg.obs_type=='pixels':
                        action = self.agent.act(time_step.observation['pixels'],
                                            goal,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                    else:    
                        action = self.agent.act(time_step.observation,
                                            goal,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                else:
                    if self.cfg.obs_type=='pixels':
                        if self.cfg.use_predictor:
                            action = self.agent.act2(time_step.observation['pixels'],
                                            meta,
                                            self.global_step,
                                            eval_mode=True,
                                            proto=self.agent)
                        else:
                            action = self.agent.act2(time_step.observation['pixels'],
                                            meta,
                                            self.global_step,
                                            eval_mode=True) 
                    else:    
                        action = self.agent.act2(time_step.observation,
                                            goal,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
            
            #save agent
            if self._global_step%200000==0:
                path = os.path.join(self.work_dir, 'optimizer_{}_{}.pth'.format(str(self.cfg.agent.name),self._global_step))
                torch.save(self.agent, path)
            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            if  self.cfg.obs_type=='pixels':
                self.replay_storage.add(time_step, meta, True)
            else:
                self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)


@hydra.main(config_path='.', config_name='pretrain')
def main(cfg):
    from pretrain import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()

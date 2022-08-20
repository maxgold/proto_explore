import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
import glob
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path
import torch.nn.functional as F
import hydra
import numpy as np
import torch
import wandb
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_replay_buffer, ndim_grid
import matplotlib.pyplot as plt
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)

def get_state_embeddings(agent, states):
    with torch.no_grad():
        s = agent.encoder(states)
        s = agent.predictor(s)
        s = agent.projector(s)
        s = F.normalize(s, dim=1, p=2)
    return s

def visualize_prototypes_visited(agent, replay_dir, cfg, env, model_step):
    replay_buffer = make_replay_buffer(env,
            replay_dir,
            100000,
            cfg.batch_size,
            0,
            cfg.discount,
            goal=False,
            relabel=False,
            replay_dir2 = False,
            obs_type=cfg.obs_type,
            model_step=model_step
            )
    pix, states, actions = replay_buffer._sample(eval_pixel=True)
    if states == '':
        print('nothing in buffer yet')
    else:
        pix = pix.astype(np.float64)
        states = states.astype(np.float64)
        states = states.reshape(-1,4)
        grid = pix.reshape(-1,9, 84, 84)
        grid = torch.tensor(grid).cuda().float()
        grid = get_state_embeddings(agent, grid)
        return grid, states
        



class Workspace:
    def __init__(self, cfg, agent):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.agent_path = agent
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type,
                str(cfg.seed)
            ])
            wandb.init(project="urlb", group=cfg.agent.name, name=exp_name)

        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        task = PRIMAL_TASKS[self.cfg.domain]
        self.train_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)
        self.eval_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, cfg.seed)

        # create agent
        #import IPython as ipy; ipy.embed(colors='neutral')
        self.agent = torch.load(self.agent_path)
        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer2')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount, False,
                                                cfg.obs_type)
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

    def eval(self, replay_dir, model, replay_dir2):
        print('visualizing proto')
        grid_embeddings = torch.empty(int(int(model)*.25), 512)
        states = torch.empty(int(int(model)*.25), 4)
        for i in range(int(int(model)*.25)):
            grid, state = visualize_prototypes_visited(self.agent, replay_dir, self.cfg, self.eval_env, model)
            grid_embeddings[i] = grid
            states[i] = torch.tensor(state).cuda().float()
        
        protos = self.agent.protos.weight.data.detach().clone()
        protos = F.normalize(protos, dim=1, p=2)
        dist_mat = torch.cdist(protos, grid_embeddings)
        closest_points = dist_mat.argmin(-1)
        proto2d = states[closest_points.cpu(), :2]
        
        plt.clf()
        fig, ax = plt.subplots()
        ax.scatter(proto2d[:,0], proto2d[:,1])
        plt.savefig(f"./{self._global_step}_proto2d_eval.png")

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
        if self.cfg.obs_type == 'pixels':
        
            self.replay_storage.add(time_step, meta,True)
        else:
            self.replay_storage.add(time_step, meta)

        #self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                #self.train_video_recorder.save(f'{self.global_frame}.mp4')
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
                if self.cfg.obs_type =='pixels':
                    self.replay_storage.add(time_step, meta,True)
                else:
                    self.replay_storage.add(time_step, meta)
                #self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step) and self.global_step!=0:
                print('evaluating')
                self.logger.log('eval_total_time', self.timer.total_time(),
                        self.global_frame)
                #self.eval()

            meta = self.agent.update_meta(meta, self.global_step, time_step)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                if self.cfg.obs_type == 'pixels':

                    action = self.agent.act(time_step.observation['pixels'].copy(),
                                            meta,
                                            self.global_step,
                                            eval_mode=False)
                else:
                    action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
            
            if self._global_step%100000==0:
                path = os.path.join(self.work_dir, 'optimizer_{}_{}.pth'.format(str(self.cfg.agent.name),self._global_step))
                torch.save(self.agent, path)

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            if self.cfg.obs_type == 'pixels':
                self.replay_storage.add(time_step, meta,True)
            else:
                self.replay_storage.add(time_step, meta)
            #self.train_video_recorder.record(time_step.observation)
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
    from eval_pixel import Workspace as W
    root_dir = Path.cwd()
    agents = glob.glob(str(cfg.path)+'/optimizer_proto_100000.pth')
    print(agents)

    for ix, x in enumerate(agents):
        
        workspace = W(cfg, x)
        model = str(x).split('_')[-1]
        model = str(model).split('.')[-2]
        replay_dir = Path(cfg.replay_dir)
        if cfg.replay_dir2:
            replay_dir2 = Path(cfg.replay_dir2)
        else:
            replay_dir2 = False
        print('model_step', model)
        workspace.eval(replay_dir, model, replay_dir2)
        print(ix)

if __name__ == '__main__':
    main()

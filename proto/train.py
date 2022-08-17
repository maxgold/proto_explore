import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

import copy
import math
import pickle as pkl
import sys
import time

import numpy as np
from pathlib import Path
import dmc
import hydra
import torch
import utils
from logger import Logger, save
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_replay_buffer
from video import VideoRecorder
from dmc_benchmark import PRIMAL_TASKS
torch.backends.cudnn.benchmark = True


def get_state_embeddings(agent, states):
    with torch.no_grad():
        s = agent.encoder(states)
        s = agent.predictor(s)
        s = agent.projector(s)
        s = F.normalize(s, dim=1, p=2)
    return s

def visualize_prototypes_visited(agent, work_dir, cfg, env):
    replay_dir = work_dir / 'buffer2' / 'buffer_copy'
    replay_buffer = make_replay_buffer(env,
                                        replay_dir,
                                        1000000,
                                        cfg.batch_size,
                                        0,
                                        cfg.discount,
                                        goal=False,
                                        relabel=False,
                                        replay_dir2 = False
                                        )
    states, actions = replay_buffer.parse_dataset()
    if states == '':
        print('nothing in buffer yet')
    else:
        states = states.astype(np.float64)
        grid = states.reshape(-1,4)
        grid = torch.tensor(grid).cuda().float()
        grid_embeddings = get_state_embeddings(agent, grid)
        protos = agent.protos.weight.data.detach().clone()
        protos = F.normalize(protos, dim=1, p=2)
        dist_mat = torch.cdist(protos, grid_embeddings)
        closest_points = dist_mat.argmin(-1)
        return grid[closest_points, :2].cpu()

class Workspace(object):
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')
        self.model_dir = utils.make_dir(self.work_dir, 'model')
        self.buffer_dir = utils.make_dir(self.work_dir, 'buffer2')

        self.cfg = cfg
        self.logger = Logger(self.work_dir,
                            use_tb=self.cfg.use_tb,
                            use_wandb=self.cfg.use_wandb)

        utils.set_seed_everywhere(self.cfg.seed)
        self.device = torch.device(self.cfg.device)
        
        # create envs
        try:
            task = PRIMAL_TASKS[self.cfg.domain]
        except:
            task = self.cfg.domain
        
        self.env = dmc.make(task, self.cfg.obs_type, self.cfg.frame_stack, self.cfg.action_repeat,
                            self.cfg.seed)
        self.eval_env = dmc.make(task, self.cfg.obs_type, self.cfg.frame_stack, self.cfg.action_repeat,
                                 self.cfg.seed + 1)

       
        obs_spec = self.env.observation_spec()['pixels']
        action_spec = self.env.action_spec()

        cfg.agent.params.obs_shape = obs_spec.shape
        cfg.agent.params.action_shape = action_spec.shape
        cfg.agent.params.action_range = [
            float(action_spec.minimum.min()),
            float(action_spec.maximum.max())
        ]

        # exploration agent uses intrinsic reward
        self.expl_agent = hydra.utils.instantiate(cfg.agent,
                                                  task_agnostic=True)
        # task agent uses extr extrinsic reward
        self.task_agent = hydra.utils.instantiate(cfg.agent,
                                                  task_agnostic=False)
        self.task_agent.assign_modules_from(self.expl_agent)
        
        data_specs = (self.env.observation_spec(),
                        self.env.action_spec(),
                        specs.Array((1,), np.float32, 'reward'),
                        specs.Array((1,), np.float32, 'discount'))


        #if cfg.load_pretrained:
        #    pretrained_path = utils.find_pretrained_agent(
        #        cfg.pretrained_dir, cfg.env, cfg.seed, cfg.pretrained_step)
        #    print(f'snapshot is taken from: {pretrained_path}')
        #    pretrained_agent = utils.load(pretrained_path)
        #    self.task_agent.assign_modules_from(pretrained_agent)

        # storage for the task-agnostic phase
        self.expl_buffer = ReplayBufferStorage(data_specs,
                                                self.work_dir / 'buffer1')
        # storage for task-specific phase
        self.task_buffer = ReplayBufferStorage(data_specs,
                                                self.work_dir / 'buffer2')
        # create replay buffer
        self.replay_loader1 = make_replay_loader(self.expl_buffer,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount, False)

        self.replay_loader2  = make_replay_loader(self.task_buffer,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount, False)

        self._replay_iter1 = None
        self._replay_iter2 = None


        self.eval_video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.step = 0

    @property
    def replay_iter1(self):
        if self._replay_iter1 is None:
            self._replay_iter1 = iter(self.replay_loader1)
        return self._replay_iter1

    @property
    def replay_iter2(self):
        if self._replay_iter2 is None:
            self._replay_iter2 = iter(self.replay_loader2)
        return self._replay_iter2

    

    def get_agent(self):
        if self.step < self.cfg.num_expl_steps:
            return self.expl_agent
        return self.task_agent

    def get_buffer(self):
        if self.step < self.cfg.num_expl_steps:
            return self.replay_iter1
        return self.replay_iter2

    def evaluate(self):
        if self.step % int(1e4) == 0:
            proto2d = visualize_prototypes_visited(self.agent, self.work_dir, self.cfg, self.eval_env)
            plt.clf()
            fig, ax = plt.subplots()
            ax.scatter(proto2d[:,0], proto2d[:,1])
            plt.savefig(f"./{self._global_step}_proto2d_eval.png")

    def run(self):
        
        episode, episode_reward, episode_step = 0, 0, 0
        start_time = time.time()
        done = True
        while self.step <= self.cfg.num_train_steps:
            if done:
                if self.step > 0:
                    fps = episode_step / (time.time() - start_time)
                    self.logger.log('train/fps', fps, self.step)
                    start_time = time.time()
                    self.logger.log('train/episode_reward', episode_reward, self.step)
                    self.logger.log('train/episode', episode, self.step)
                    self.logger.dump(self.step, ty='train')

                time_step = self.env.reset()
                obs = time_step.observation['pixels']
                episode_reward = 0
                episode_step = 0
                episode += 1

                self.logger.log('train/episode', episode, self.step)

            agent = self.get_agent()
            replay_buffer = self.get_buffer()
            # evaluate agent periodically
            if self.step % self.cfg.eval_frequency == 0 and self.step!=0:
                self.logger.log('eval/episode', episode - 1, self.step)
                #self.evaluate()

            # save agent periodically
            if self.cfg.save_model and self.step % self.cfg.save_frequency == 0:
                utils.save(
                    self.expl_agent,
                    os.path.join(self.model_dir, f'expl_agent_{self.step}.pth'))
                utils.save(
                    self.task_agent,
                    os.path.join(self.model_dir, f'task_agent_{self.step}.pth'))
            #if self.cfg.save_buffer and self.step % self.cfg.save_frequency == 0:
            #    replay_buffer.save(self.buffer_dir, self.cfg.save_pixels)
            # sample action for data collection
            if self.step < self.cfg.num_random_steps:
                spec = self.env.action_spec()
                action = np.random.uniform(spec.minimum, spec.maximum,
                                           spec.shape)
            else:
                with utils.eval_mode(agent):
                    action = agent.act(obs, sample=True)

            agent.update(self.replay_iter, self.step)

            time_step = self.env.step(action)
            next_obs = time_step.observation['pixels']

            # allow infinite bootstrap
            done = time_step.last()
            episode_reward += time_step.reward

            self.replay_storage.add(time_step,True)

            obs = next_obs
            episode_step += 1
            self.step += 1


@hydra.main(config_path='.', config_name='config')
def main(cfg):
    from train import Workspace as W
    workspace = W(cfg)
    workspace.run()


if __name__ == '__main__':
    main()

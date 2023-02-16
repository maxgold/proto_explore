import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['HYDRA_FULL_ERROR'] = '1'

import seaborn as sns;
from logger import Logger

sns.set_theme()
import hydra
import numpy as np
import torch
from dm_env import specs
import pandas as pd
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
#from agent_utils import *
#from eval_ops import *
from agent.expert import ExpertAgent
from pathlib import Path
import utils
import dmc
torch.backends.cudnn.benchmark = True
import wandb

def make_agent(obs_type, obs_spec, action_spec, goal_shape, num_expl_steps, cfg, lr=.0001, hidden_dim=1024,
               num_protos=512,
               update_gc=2, gc_only=False, offline=False, tau=.1, num_iterations=3, feature_dim=50, pred_dim=128,
               proj_dim=512,
               batch_size=1024, update_proto_every=10, lagr=.2, margin=.5, lagr1=.2, lagr2=.2, lagr3=.3,
               stddev_schedule=.2,
               stddev_clip=.3, update_proto=2, stddev_schedule2=.2, stddev_clip2=.3, update_enc_proto=False,
               update_enc_gc=False, update_proto_opt=True,
               normalize=False, normalize2=False):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    cfg.goal_shape = goal_shape
    cfg.lr = lr
    cfg.hidden_dim = hidden_dim
    cfg.num_protos = num_protos
    cfg.tau = tau

    if cfg.name.startswith('proto'):
        cfg.update_gc = update_gc
    cfg.offline = offline
    cfg.gc_only = gc_only
    cfg.batch_size = batch_size
    cfg.tau = tau
    cfg.num_iterations = num_iterations
    cfg.feature_dim = feature_dim
    cfg.pred_dim = pred_dim
    cfg.proj_dim = proj_dim
    cfg.lagr = lagr
    cfg.margin = margin
    cfg.stddev_schedule = stddev_schedule
    cfg.stddev_clip = stddev_clip
    if cfg.name == 'protox':
        cfg.lagr1 = lagr1
        cfg.lagr2 = lagr2
        cfg.lagr3 = lagr3

    cfg.update_proto_every = update_proto_every
    cfg.stddev_schedule2 = stddev_schedule2
    cfg.stddev_clip2 = stddev_clip2
    cfg.update_enc_proto = update_enc_proto
    cfg.update_enc_gc = update_enc_gc
    cfg.update_proto_opt = update_proto_opt
    cfg.normalize = normalize
    cfg.normalize2 = normalize2
    print('shape', obs_spec.shape)
    return hydra.utils.instantiate(cfg)

def ndim_grid(ndims, space):
    L = [np.linspace(-.29,.29,space) for i in range(ndims)]
    return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T

class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        work_path = str(os.getcwd().split('/')[-2]) + '/' + str(os.getcwd().split('/')[-1])

        # create logger
        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, cfg.task, cfg.obs_type,
                str(cfg.seed), str(cfg.tmux_session), work_path
            ])
            wandb.init(project="urlb1", group=cfg.agent.name, name=exp_name)

        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)

        # create envs

        task = self.cfg.task
        self.pmm = False
        if self.cfg.task.startswith('point_mass'):
            self.pmm = True


        # two different routes for pmm vs. non-pmm envs
        # TODO
        # write into function: init_envs

        self.no_goal_task = self.cfg.task_no_goal
        idx = np.random.randint(0, 400)
        goal_array = ndim_grid(2, 20)
        self.first_goal = np.array([goal_array[idx][0], goal_array[idx][1]])

        self.train_env1 = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                    cfg.action_repeat, seed=None, goal=self.first_goal)
        print('goal', self.first_goal)

        self.train_env_no_goal = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                            cfg.action_repeat, seed=None, goal=None)
        print('no goal task env', self.no_goal_task)

        self.train_env = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                    cfg.action_repeat, seed=None)

        self.eval_env = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                    cfg.action_repeat, seed=None, goal=self.first_goal)

        self.eval_env_no_goal = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                            cfg.action_repeat, seed=None, goal=None)

        self.eval_env_goal = dmc.make(self.no_goal_task, 'states', cfg.frame_stack,
                                        1, seed=None, goal=None)


        if cfg.cassio:
            self.pwd = '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark'
        elif cfg.greene:
            self.pwd = '/vast/nm1874/dm_control_2022/proto_explore/url_benchmark'
        elif cfg.pluto:
            self.pwd = '/home/nina/proto_explore/url_benchmark'

        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                (3 * self.cfg.frame_stack, 84, 84),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg.agent,
                                cfg.lr,
                                cfg.hidden_dim,
                                cfg.num_protos,
                                cfg.update_gc,
                                False,
                                cfg.offline,
                                cfg.tau,
                                cfg.num_iterations,
                                cfg.feature_dim,
                                cfg.pred_dim,
                                cfg.proj_dim,
                                batch_size=cfg.batch_size,
                                lagr=cfg.lagr,
                                margin=cfg.margin,
                                stddev_schedule=cfg.stddev_schedule,
                                stddev_clip=cfg.stddev_clip,
                                stddev_schedule2=cfg.stddev_schedule2,
                                stddev_clip2=cfg.stddev_clip2,
                                update_enc_proto=cfg.update_enc_proto,
                                update_enc_gc=cfg.update_enc_gc,
                                update_proto_opt=cfg.update_proto_opt,
                                normalize=cfg.normalize,
                                normalize2=cfg.normalize2)

        self.agent = ExpertAgent()
        # get meta specs

        #meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage

        self.replay_storage = ReplayBufferStorage(data_specs, None,
                                                  self.work_dir / 'buffer1')

        lst=[]
        goal_array = ndim_grid(2,20)
        for ix,x in enumerate(goal_array):
            if (-.02<x[0]  or  x[1]<.02):
                lst.append(ix)


        goal_array=np.delete(goal_array, lst,0)
        print(len(goal_array))
        
        for ix in range(100):

            init_state = np.random.uniform(.25,.29,size=(2,))
            init_state[0] = init_state[0]*(-1)

            for x in goal_array:

                done=False
                goal_state = x
                episode_step = 0
                episode_reward = 0

                with self.eval_env_no_goal.physics.reset_context():
                    self.eval_env_no_goal.physics.set_state(np.array([goal_state[0], goal_state[1],0,0]))

                goal_pix = self.eval_env_no_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
                goal_pix = np.transpose(goal_pix, (2,0,1))
                goal_pix = np.tile(goal_pix, (cfg.frame_stack,1,1))

                self.train_env1 = dmc.make(cfg.task, cfg.obs_type,
                                                    cfg.frame_stack,cfg.action_repeat,
                                                    seed=None, goal=goal_state, init_state=init_state)

                time_step = self.train_env1.reset()
                self.train_env_no_goal = dmc.make(cfg.task_no_goal, cfg.obs_type, cfg.frame_stack,
                                                cfg.action_repeat, seed=None, goal=None,
                                                init_state=time_step.observation['observations'][:2])

                time_step_no_goal = self.train_env_no_goal.reset()

                if ((time_step.last() and self.actor1) or (time_step.last() and self.actor)) and (
                            self.global_step != self.switch_gc or self.cfg.model_path):
                        print('last')
                        self._global_episode += 1



                        self.replay_storage.add_goal_general(time_step, self.train_env1.physics.get_state(), None,
                                                                goal_pix, goal_state,
                                                                time_step_no_goal, True, last=True, expert=True)

                        episode_step = 0
                        episode_reward = 0
                
                while not time_step.last():
                    action = self.agent.act(time_step.physics, goal_state, None, None)
                    action = np.array(action,dtype="float32")
                    #action = np.clip(action, -1,1)
                    time_step = self.train_env1.step(action)
                    time_step_no_goal = self.train_env_no_goal.step(action)
                    self.replay_storage.add_goal_general(time_step, self.train_env1.physics.get_state(), None,
                                                            goal_pix, goal_state,
                                                            time_step_no_goal, True, expert=True)
                    episode_step+=1
                    episode_reward+=time_step.reward
                    norm = np.linalg.norm(action-np.array([0,0]))
                    #if episode_reward > 100:
                    #    print('episode', episode_step)
                    #    print('rewar', episode_reward)
                    #    print('act', action)
                    #    self.replay_storage.add_goal_general(time_step, self.train_env1.physics.get_state(), None,
                    #                                                                      goal_pix, goal_state,
                    #                                                                                                                                                                                                                    time_step_no_goal, True, expert=True, last=True)
                    #    done=True
                    #    break

                #if done:
                #    continue
@hydra.main(config_path='.', config_name='pretrain')
def main(cfg):
    from expert_buffer import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()




import warnings


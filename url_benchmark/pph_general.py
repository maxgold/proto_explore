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
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_replay_buffer
from video import TrainVideoRecorder, VideoRecorder
from agent_utils import *
from eval_ops import *
from pathlib import Path
import gym

torch.backends.cudnn.benchmark = True


def make_agent(obs_type, obs_spec, action_spec, goal_shape, num_expl_steps, cfg, lr=.0001, hidden_dim=1024, num_protos=512,
               update_gc=2, tau=.1, num_iterations=3, feature_dim=50, pred_dim=128, proj_dim=512,
               batch_size=1024, update_proto_every=10, stddev_schedule=.2, stddev_clip=.3, update_proto=2, 
               stddev_schedule2=.2, stddev_clip2=.3, update_enc_proto=False, update_enc_gc=False, update_proto_opt=True,
               normalize=False, normalize2=False, sl=False, encoder1=False, encoder2=False, encoder3=False, feature_dim_gc=50, inv=False,
               use_actor_trunk=False, use_critic_trunk=False, init_from_proto=False, init_from_ddpg=False, pretrained_feature_dim=16,
               scale=None, gc_inv=False):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    cfg.goal_shape = goal_shape
    cfg.lr = lr
    cfg.hidden_dim = hidden_dim

    if cfg.name.startswith('proto'):
        cfg.num_protos = num_protos
        cfg.feature_dim = feature_dim
        cfg.update_gc = update_gc
        cfg.tau = tau
        cfg.num_iterations = num_iterations
        cfg.pred_dim = pred_dim
        cfg.proj_dim = proj_dim
        cfg.normalize = normalize
        cfg.normalize2 = normalize2
        cfg.update_proto_every = update_proto_every
        cfg.update_enc_proto = update_enc_proto
        cfg.update_enc_gc = update_enc_gc
        cfg.update_proto_opt = update_proto_opt
        cfg.gc_inv = inv

    cfg.batch_size = batch_size
    cfg.feature_dim = feature_dim
    cfg.stddev_schedule = stddev_schedule
    cfg.stddev_clip = stddev_clip
    cfg.stddev_schedule2 = stddev_schedule2
    cfg.stddev_clip2 = stddev_clip2
    cfg.sl = sl
    cfg.encoder1 = encoder1
    cfg.encoder2 = encoder2
    cfg.encoder3 = encoder3
    cfg.feature_dim_gc = feature_dim_gc
    cfg.inv = inv
    cfg.use_actor_trunk = use_actor_trunk
    cfg.use_critic_trunk = use_critic_trunk
    cfg.scale = scale

    if cfg.name.startswith('ddpg'):
        cfg.init_from_proto = init_from_proto
        cfg.init_from_ddpg = init_from_ddpg
        cfg.pretrained_feature_dim = pretrained_feature_dim

    return hydra.utils.instantiate(cfg)


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

        if self.pmm:
            if self.cfg.velocity_control:
                assert self.cfg.stddev_schedule <= .02 and self.cfg.stddev_schedule2 <= .02
                assert self.cfg.stddev_clip <= .02 and self.cfg.stddev_clip2 <= .02
                assert self.cfg.scale is not None and self.cfg.frame_stack == 1

            self.no_goal_task = self.cfg.task_no_goal
            idx = np.random.randint(0, 400)
            goal_array = ndim_grid(2, 20)
            self.first_goal = np.array([goal_array[idx][0], goal_array[idx][1]])

            self.train_env1 = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                       cfg.action_repeat, seed=None, goal=self.first_goal, camera_id=cfg.camera_id)
            print('goal', self.first_goal)

            self.train_env_no_goal = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                              cfg.action_repeat, seed=None, goal=None, camera_id=cfg.camera_id)
            print('no goal task env', self.no_goal_task)

            self.train_env = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                      cfg.action_repeat, seed=None, camera_id=cfg.camera_id)

            self.eval_env = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                     cfg.action_repeat, seed=None, goal=self.first_goal, camera_id=cfg.camera_id)

            self.eval_env_no_goal = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                             cfg.action_repeat, seed=None, goal=None, camera_id=cfg.camera_id)

            self.eval_env_goal = dmc.make(self.no_goal_task, 'states', cfg.frame_stack,
                                          1, seed=None, goal=None, camera_id=cfg.camera_id)
        else:
            self.train_env1 = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                       cfg.action_repeat, seed=None, camera_id=cfg.camera_id)
            self.train_env = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                      cfg.action_repeat, seed=None, camera_id=cfg.camera_id)
            self.eval_env = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                     cfg.action_repeat, seed=None, camera_id=cfg.camera_id)

        if cfg.cassio:
            self.pwd = '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark'
        elif cfg.greene:
            self.pwd = '/vast/nm1874/dm_control_2022/proto_explore/url_benchmark'
        elif cfg.pluto or cfg.irmak:
            self.pwd = '/home/nina/proto_explore/url_benchmark'
        else:
            self.pwd = None
            # create agent

        if cfg.agent.name == 'proto':
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
                                    cfg.tau,
                                    cfg.num_iterations,
                                    cfg.feature_dim,
                                    cfg.pred_dim,
                                    cfg.proj_dim,
                                    batch_size=cfg.batch_size,
                                    stddev_schedule=cfg.stddev_schedule,
                                    stddev_clip=cfg.stddev_clip,
                                    stddev_schedule2=cfg.stddev_schedule2,
                                    stddev_clip2=cfg.stddev_clip2,
                                    update_enc_proto=cfg.update_enc_proto,
                                    update_enc_gc=cfg.update_enc_gc,
                                    update_proto_opt=cfg.update_proto_opt,
                                    normalize=cfg.normalize,
                                    normalize2=cfg.normalize2,
                                    sl = cfg.sl,
                                    encoder1 = cfg.encoder1,
                                    encoder2 = cfg.encoder2,
                                    encoder3 = cfg.encoder3,
                                    feature_dim_gc = cfg.feature_dim_gc,
                                    inv = cfg.inv,
                                    use_actor_trunk=cfg.use_actor_trunk,
                                    use_critic_trunk=cfg.use_critic_trunk,
                                    scale = cfg.scale,
                                    gc_inv=cfg.inv)

        elif cfg.agent.name == 'ddpg':
            self.agent = make_agent(cfg.obs_type,
                                    self.train_env.observation_spec(),
                                    self.train_env.action_spec(),
                                    (3 * self.cfg.frame_stack, 84, 84),
                                    cfg.num_seed_frames // cfg.action_repeat,
                                    cfg.agent,
                                    cfg.lr,
                                    cfg.hidden_dim,
                                    cfg.update_gc,
                                    batch_size=cfg.batch_size,
                                    feature_dim=cfg.feature_dim,
                                    stddev_schedule=cfg.stddev_schedule,
                                    stddev_clip=cfg.stddev_clip,
                                    stddev_schedule2=cfg.stddev_schedule2,
                                    stddev_clip2=cfg.stddev_clip2,
                                    sl = cfg.sl,
                                    encoder1 = cfg.encoder1,
                                    encoder2 = cfg.encoder2,
                                    encoder3 = cfg.encoder3,
                                    feature_dim_gc = cfg.feature_dim_gc,
                                    inv = cfg.inv,
                                    use_actor_trunk=cfg.use_actor_trunk,
                                    use_critic_trunk=cfg.use_critic_trunk,
                                    init_from_ddpg=cfg.init_from_ddpg,
                                    init_from_proto=cfg.init_from_proto,
                                    pretrained_feature_dim=cfg.pretrained_feature_dim)

        # initialize from pretrained
        print('pwd', self.pwd)
        print('model p', cfg.model_path)
        if cfg.model_path:
            assert os.path.isfile(self.pwd + cfg.model_path)
            self.pretrained_agent = torch.load(self.pwd + cfg.model_path)
            if self.cfg.resume_training:
                self.agent.init_from(self.pretrained_agent)
            elif self.cfg.gc_only:
                print('gc only, only loading in the encoder from pretrained agent')
                if self.cfg.sl:
                    self.agent.init_encoder_from(self.pretrained_agent.encoder)
                elif self.cfg.init_from_proto:
                    self.agent.init_encoder_trunk_from(self.pretrained_agent.encoder, self.pretrained_agent.critic2, self.pretrained_agent.actor2)
                elif self.cfg.init_from_ddpg:
                    self.agent.init_encoder_trunk_gc_from(self.pretrained_agent.encoder, self.pretrained_agent.critic, self.pretrained_agent.actor)
            path = self.cfg.model_path.split('/')
            path = Path(self.pwd + '/'.join(path[:-1]))

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage1 = ReplayBufferStorage(data_specs, meta_specs,
                                                   self.work_dir / 'buffer1')
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer2')
        self.replay_storage_eval = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer_eval')

        # create replay buffer
        self.combine_storage = False
        self.combine_storage_gc = False
        if self.cfg.combine_storage_gc:
            replay_dir2_gc = self.work_dir / 'buffer2' / 'buffer_copy'
            replay_dir2 = None
            self.combine_storage_gc = True

        elif self.cfg.resume_training:
            replay_dir2_gc = path / 'buffer1' / 'buffer_copy'
            replay_dir2 = path / 'buffer2' / 'buffer_copy'
            self.combine_storage_gc = True
            self.combine_storage = True
        else:
            replay_dir2_gc = None
            replay_dir2 = None
            
        if self.cfg.expert_buffer and self.cfg.offline_gc:
            #first path lots of no action
            print('buffer', self.cfg.buffer_num)
            if self.cfg.greene:
                if self.cfg.buffer_num == 0:
                    buffer_path = Path('/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/exp_local/2023.02.06/151622_proto_encoder1/buffer1/buffer_copy')
                elif self.cfg.buffer_num == 1:
                    buffer_path = Path('/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/exp_local/2023.02.10/123706_proto_sl_inv/buffer1/buffer_copy')
            
                #early stopping, init state all = -.29, .29, ndim grid (2,10)
                #2023.02.11/174359_proto_sl_inv/

                #no early stopping, init state all = -.29, .29, ndim (2, 10)
                #2023.02.11/180801_proto_sl_inv/
                elif self.cfg.buffer_num == 2:
                    buffer_path = Path('/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/exp_local/2023.02.11/180801_proto_sl_inv/buffer1/buffer_copy')

                #no early stopping, init = -.29, .29, ndim(2,20)
                elif self.cfg.buffer_num == 3:
                    buffer_path = Path('/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/exp_local/2023.02.11/200427_proto_sl_inv/buffer1/buffer_copy')
            
                #early stopping, init=-.29,.29, ndim(2,20)
                elif self.cfg.buffer_num == 4:
                    buffer_path = Path('/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/exp_local/2023.02.13/125305_proto_sl_inv/buffer1/buffer_copy')
 
                #first path lots of no action
            elif self.cfg.cassio:
                buffer_path = Path('/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2023.02.15/234008_proto/buffer1/buffer_copy')

            elif self.cfg.irmak:
                buffer_path = Path('/home/nina/proto_explore/url_benchmark/exp_local/2023.03.03/140228_proto/buffer1/buffer_copy')
                #if self.cfg.buffer_num == 0:
                #    buffer_path = Path('/home/nina/proto_explore/url_benchmark/exp_local/2023.02.14/163804_proto_sl_inv/buffer1/buffer_copy')
                #elif self.cfg.buffer_num == 1:
                #    buffer_path = Path('/home/nina/proto_explore/url_benchmark/exp_local/2023.02.14/163804_proto_sl_inv/buffer1/buffer_copy')

        elif self.cfg.offline_gc:
            if self.cfg.init_from_proto:
                buffer_path = path / 'buffer2' / 'buffer_copy'
            else:
                buffer_path = self.work_dir / 'buffer2' / 'buffer_copy'
            print('none expert buffer', buffer_path)
        # TODO
        # figure out why files "disappear" in buffer_copy when used by another loader
        # figure out why we can't add parse data function to data loader
        if self.cfg.offline_gc:
            print('offline buffer')
            print('model path', self.cfg.model_path)
            if self.cfg.model_path is False:
                self.cfg.offline_model_step = None
                self.cfg.offline_model_step_lb = None
                print('offline model step', self.cfg.offline_model_step)
                print('offline model step lb', self.cfg.offline_model_step_lb)

            self.replay_loader1 = make_replay_buffer(
                                                    buffer_path,
                                                    cfg.replay_buffer_gc,
                                                    cfg.batch_size_gc,
                                                    cfg.replay_buffer_num_workers,
                                                    cfg.discount,
                                                    offset=100,
                                                    goal=True,
                                                    relabel=False,
                                                    replay_dir2=False,
                                                    obs_type=self.cfg.obs_type,
                                                    offline=False,
                                                    nstep=self.cfg.nstep,
                                                    eval=False,
                                                    inv=cfg.inv,
                                                    goal_offset=self.cfg.goal_offset,
                                                    pmm=self.pmm,
                                                    model_step=self.cfg.offline_model_step,
                                                    model_step_lb=self.cfg.offline_model_step_lb,
                                                    reverse=self.cfg.reverse)
        else:
            self.replay_loader1 = make_replay_loader(self.replay_storage1,
                                                    self.combine_storage_gc,
                                                    cfg.replay_buffer_gc,
                                                    cfg.batch_size_gc,
                                                    cfg.replay_buffer_num_workers,
                                                    False, cfg.nstep1, cfg.discount,
                                                    True, cfg.hybrid_gc, cfg.obs_type,
                                                    cfg.hybrid_pct, sl=cfg.sl,
                                                    replay_dir2=replay_dir2_gc,
                                                    loss=cfg.loss_gc, test=cfg.test,
                                                    tile=cfg.frame_stack,
                                                    pmm=self.pmm,
                                                    obs_shape=self.train_env1.physics.state().shape[0],
                                                    general=True,
                                                    inv=cfg.inv,
                                                    goal_offset=self.cfg.goal_offset)

        self.replay_loader = make_replay_loader(self.replay_storage,
                                                self.combine_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep2, cfg.discount,
                                                goal=False,
                                                obs_type=cfg.obs_type,
                                                replay_dir2=replay_dir2,
                                                loss=cfg.loss,
                                                test=cfg.test
                                                )

        self._replay_iter = None
        self._replay_iter1 = None

        if self.cfg.egocentric is False:
            self.video_recorder = VideoRecorder(
                self.work_dir if cfg.save_video else None,
                camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
                use_wandb=self.cfg.use_wandb)
            self.train_video_recorder = TrainVideoRecorder(
                self.work_dir if cfg.save_train_video else None,
                camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
                use_wandb=self.cfg.use_wandb)
        else:
            self.video_recorder = VideoRecorder(
                self.work_dir if cfg.save_video else None,
                camera_id=1 if 'quadruped' not in self.cfg.domain else 2,
                use_wandb=self.cfg.use_wandb)
            self.train_video_recorder = TrainVideoRecorder(
                self.work_dir if cfg.save_train_video else None,
                camera_id=1 if 'quadruped' not in self.cfg.domain else 2,
                use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.loaded = False
        self.loaded_uniform = False
        self.uniform_goal = []
        self.uniform_state = []
        self.count_uniform = 0
        self.goal_loaded = False
        self.distance_goal = []
        self.count = 0
        self.global_success_rate = []
        self.global_index = []
        self.storage1 = False
        self.distance_goal_init = {}
        self.goal_loaded = False

        if self.cfg.proto_goal_intr:
            self.dim = 10
        else:
            if self.cfg.gc_only:
                self.dim = 20
                #TODO: double check this from pretrain_gc_only.py
            else:
                self.dim = self.agent.protos.weight.data.shape[0]

        # clean up this section
        self.proto_goals = np.zeros((self.dim, 3 * self.cfg.frame_stack, 84, 84))
        self.proto_goals_state = np.zeros((self.dim, self.train_env.physics.get_state().shape[0]))
        self.proto_goals_dist = np.zeros((self.dim, 1))
        self.proto_goals_matrix = np.zeros((60, 60))
        self.proto_goals_id = np.zeros((self.dim, 2))
        self.actor = True
        self.actor1 = False
        self.final_df = pd.DataFrame(columns=['avg', 'med', 'max', 'q7', 'q8', 'q9'])
        self.reached_goals = np.zeros((2, 60, 60))
        self.origin = None
        self.proto_explore_count = 0
        self.previous_matrix = None
        self.current_matrix = None
        self.count = 0
        self.unreached_goals = np.empty((0, self.train_env.physics.get_state().shape[0]))
        self.proto_last_explore = 0
        self.current_init = np.empty((0, self.train_env.physics.get_state().shape[0]))
        self.eval_reached = np.empty((0, self.train_env.physics.get_state().shape[0]))
        self.unreached = False
        self.reinitiate = False
        self.gc_init = False
        self.gc_step = 0 
        self.proto_step = 0
        self.proto_uniformness = np.empty((0,1))
        self.num_reached = np.empty((0,1))

        self.switch_gc = self.cfg.switch_gc

        if self.cfg.gc_only or self.cfg.proto_only:
            self.hybrid=False
        else:
            self.hybrid=True

    

        # if both proto_only & gc_only are true, alert wrong config
        if all((self.cfg.proto_only, self.cfg.gc_only)):
            assert NotImplementedError

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
    def replay_iter1(self):
        if self._replay_iter1 is None:
            self._replay_iter1 = iter(self.replay_loader1)
        return self._replay_iter1

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def evaluate(self, eval=False):
        self.logger.log('eval_total_time', self.timer.total_time(),
                        self.global_frame)
        if self.cfg.gc_only and self.cfg.offline_gc is False:
            self.proto_goals, self.proto_goals_state, self.proto_goals_dist = eval_proto_gc_only(self.cfg, self.agent, self.device, self.pwd, self.global_step, self.pmm, self.train_env, self.proto_goals, self. proto_goals_state, self.proto_goals_dist, self.dim, self.work_dir, self.current_init, self.replay_storage1.state_visitation_gc, self.replay_storage1.reward_matrix, self.replay_storage1.goal_state_matrix, self.replay_storage.state_visitation_proto, self.proto_goals_matrix, self.mov_avg_5, self.mov_avg_10, self.mov_avg_20, self.mov_avg_50, self.r_mov_avg_5, self.r_mov_avg_10, self.r_mov_avg_20, self.r_mov_avg_50, eval=eval)
            eval_pmm(self.cfg, self.agent, self.eval_reached, self.video_recorder, self.global_step, self.global_frame, self.work_dir, self.replay_storage_eval)

        elif self.cfg.gc_only and self.cfg.offline_gc:
            #xy = np.array([.25, .25, .1, .1])
            #if self.cfg.debug and self.cfg.eval_proto_goals is False:
            #    self.eval_reached = np.empty((16,4))
             #   for i in range(4):
             #       for j in range(4):
             #           self.eval_reached[i*4+j,0]=xy[i] * (-1)**i
             #           self.eval_reached[i*4+j,1]=xy[j] * (-1)**j
             #           self.eval_reached[i*4+j,2]=0
             #           self.eval_reached[i*4+j,3]=0
              #  
              #  print(self.eval_reached)     
              #  eval_pmm(self.cfg, self.agent, self.eval_reached, self.video_recorder, self.global_step, self.global_frame, self.work_dir)

            if self.goal_loaded is False and self.cfg.eval_proto_goals:
                #calls eval_pmm from eval_proto
                model_step = int(re.findall('\d+', self.cfg.model_path)[-1])
                print('model_step', model_step) 
                self.current_init, self.proto_goals_state, self.proto_uniformness, self.num_reached = eval_proto(self.cfg, self.agent, self.device, self.pwd, self.global_step, self.global_frame, self.pmm, self.train_env, self.proto_goals, self. proto_goals_state, self.proto_goals_dist, self.dim, self.work_dir, self.current_init, self.replay_storage1.state_visitation_gc, self.replay_storage1.reward_matrix, self.replay_storage1.goal_state_matrix, self.replay_storage.state_visitation_proto, self.proto_goals_matrix, eval=True, video_recorder=self.video_recorder, pretrained_agent=self.pretrained_agent, model_step=model_step, replay_storage_eval=self.replay_storage_eval, proto_uniformness=self.proto_uniformness, num_reached=self.num_reached)
                assert type(self.current_init) is not tuple
                self.goal_loaded = True
                print('eval. current init', self.current_init)

            elif self.goal_loaded is True and self.cfg.eval_proto_goals:
                self.current_init, reached = eval_pmm(self.cfg, self.agent, self.current_init, self.video_recorder, self.global_step, self.global_frame, self.work_dir, goal_states=self.proto_goals_state, replay_storage_eval=self.replay_storage_eval)
                assert type(self.current_init) is not tuple
                print('eval. current init', self.current_init)
            else:
                self.eval_reached = None
                eval_pmm_stitch(self.cfg, self.agent, self.eval_reached, self.video_recorder, self.global_step, self.global_frame, self.work_dir)

        elif self.cfg.gc_only is False and self.cfg.offline_gc:
            self.current_init, self.proto_goals_state, self.proto_uniformness, self.num_reached = eval_proto(self.cfg, self.agent, self.device, self.pwd, self.global_step, self.global_frame, self.pmm, self.train_env, self.proto_goals, self.proto_goals_state, self.proto_goals_dist, self.dim, self.work_dir, self.current_init, self.replay_storage1.state_visitation_gc, self.replay_storage1.reward_matrix, self.replay_storage1.goal_state_matrix, self.replay_storage.state_visitation_proto, self.proto_goals_matrix, eval=eval, video_recorder=self.video_recorder, replay_storage_eval=self.replay_storage_eval, proto_uniformness=self.proto_uniformness, num_reached=self.num_reached)
            assert type(self.current_init) is not tuple
            print('eval. current init', self.current_init)
        else:
            #TODO: get rid of moving avg 
            self.current_init, self.proto_goals, self.proto_goals_state, self.proto_goals_dist = eval_proto(self.cfg, self.agent, self.device, self.pwd, self.global_step, self.global_frame, self.pmm, self.train_env, self.proto_goals, self. proto_goals_state, self.proto_goals_dist, self.dim, self.work_dir, self.current_init, self.replay_storage1.state_visitation_gc, self.replay_storage1.reward_matrix, self.replay_storage1.goal_state_matrix, self.replay_storage.state_visitation_proto, self.proto_goals_matrix, eval=eval, video_recorder=self.video_recorder, replay_storage_eval=self.replay_storage_eval, proto_uniformness=self.proto_uniformness, num_reached=self.num_reached)
            assert type(self.current_init) is not tuple
            print('eval. current init', self.current_init)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        log_every_step = utils.Every(self.cfg.log_every_steps)
        gc_train_until_step = utils.Until(self.cfg.num_gc_train_frames,
                                            self.cfg.action_repeat)
        proto_train_until_step = utils.Until(self.cfg.num_proto_train_frames,
                                            self.cfg.action_repeat)     
        episode_step, episode_reward = 0, 0

        time_step = self.train_env.reset()
        time_step1 = self.train_env1.reset()
        meta = self.agent.init_meta()

        if self.cfg.obs_type == 'pixels' and self.cfg.gc_only is False:
            self.replay_storage.add(time_step, self.train_env.physics.get_state(), meta, True, pmm=self.pmm)

        metrics = None
        goal_idx = 0

        if self.pmm == False:
            time_step_no_goal = None

        if self.cfg.model_path and self.cfg.offline_gc is False:
            self.evaluate(eval=True)

        # if self.cfg.debug:
        #     self.evaluate()
        while train_until_step(self.global_step):
            if self.hybrid:
                
                self.gc_step = 0 
                while proto_train_until_step(self.proto_step):
                    if time_step.last() or episode_step >= self.cfg.episode_length:
                        self._global_episode += 1
                        # wait until all the metrics schema is populated
                        if metrics is not None and self.global_step%500==0:
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

                        # if self.cfg.hack is False:
                        #     time_step = self.train_env.reset()
                        # else:
                        time_step, self.train_env, _, _, _, _, _, _, _ = get_time_step(self.cfg, self.proto_last_explore, self.cfg.gc_only, self.current_init, self.actor, self.actor1, self.pmm, train_env=self.train_env)

                        meta = self.agent.update_meta(meta, self._global_step, time_step)

                        if self.cfg.velocity_control:
                            self.replay_storage.add(time_step, self.train_env.physics.get_state(), meta, True, last=True, pmm=self.pmm, action=vel)
                        else:
                            self.replay_storage.add(time_step, self.train_env.physics.get_state(), meta, True, last=True, pmm=self.pmm)

                        # try to save snapshot
                        # TODO
                        # check or change this snapshot saving code
                        if self.global_frame in self.cfg.snapshots:
                            self.save_snapshot()

                        episode_step = 0
                        episode_reward = 0


                    meta = self.agent.update_meta(meta, self.global_step, time_step)
                    # sample action
                    with torch.no_grad(), utils.eval_mode(self.agent):

                        action = self.agent.act2(time_step.observation['pixels'],
                                                meta,
                                                self.global_step,
                                                eval_mode=True)

                    # try to update the agent
                    if not seed_until_step(self.global_step):
                        metrics = self.agent.update(self.replay_iter, self.global_step, test=self.cfg.test)
                        self.logger.log_metrics(metrics, self.global_frame, ty='train')

                    if self.cfg.velocity_control:
                        vel = action.copy()
                        action = np.zeros(2, dtype="float32")
                        self.train_env.physics.data.qvel[0] = vel[0]
                        self.train_env.physics.data.qvel[1] = vel[1]

                    # take env step
                    time_step = self.train_env.step(action)
                    episode_reward += time_step.reward

                    if self.cfg.velocity_control:
                        self.replay_storage.add(time_step, self.train_env.physics.get_state(), meta, True, pmm=self.pmm, action=vel)
                    else:
                        self.replay_storage.add(time_step, self.train_env.physics.get_state(), meta, True, pmm=self.pmm)

                    episode_step += 1
                    self._global_step += 1
                    self.proto_step += 1
                    
            self.proto_step = 0 # reset proto_step
            print('gc_step: ', self.gc_step)
            while gc_train_until_step(self.gc_step):
                if self.gc_step == (self.cfg.num_gc_train_frames//self.cfg.action_repeat)-1:
                    # try to evaluate
                    self.evaluate()

                metrics = self.agent.update(self.replay_iter1, self.global_step, actor1=True)
                self.logger.log_metrics(metrics, self.global_step, ty="train")
                if log_every_step(self.global_step):
                    elapsed_time, total_time = self.timer.reset()
                    
                    with self.logger.log_and_dump_ctx(self.global_step, ty="train") as log:
                        log("fps", self.cfg.log_every_steps / elapsed_time)
                        log("total_time", total_time)
                        log("step", self.global_step)

                self._global_step += 1
                self.gc_step += 1
            

            self.gc_step =0


            if self._global_step % 200000 == 0 and self._global_step != 0:
                print('saving agent')
                path = os.path.join(self.work_dir,
                                    'optimizer_{}_{}.pth'.format(str(self.cfg.agent.name), self._global_step))
                torch.save(self.agent, path)

                path_goal1 = os.path.join(self.work_dir,
                                          'goal_graphx_{}_{}.csv'.format(str(self.cfg.agent.name), self._global_step))
                df1 = pd.DataFrame(self.reached_goals[0])
                df1.to_csv(path_goal1, index=False)
                path_goal2 = os.path.join(self.work_dir,
                                          'goal_graphy_{}_{}.csv'.format(str(self.cfg.agent.name), self._global_step))
                df2 = pd.DataFrame(self.reached_goals[1])
                df2.to_csv(path_goal2, index=False)
                # TODO
                # think about what else we need to save

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
    from pph_general import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()

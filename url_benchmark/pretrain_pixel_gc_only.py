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
import pandas as pd
import dmc
import utils
from scipy.spatial.distance import cdist
from logger import Logger, save
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_replay_buffer, ndim_grid, make_replay_offline
import matplotlib.pyplot as plt
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS


# class DenseResidualLayer(nn.Module):

#     def __init__(self, dim):
#         super(DenseResidualLayer, self).__init__()
#         self.linear = nn.Linear(dim, dim)

#         self.apply(utils.weight_init)

#     def forward(self, x):
#         identity = x
#         out = self.linear(x)
#         out += identity
#         return out

def make_agent(self,obs_type, obs_spec, action_spec, goal_shape,num_expl_steps, goal, cfg, hidden_dim, batch_size, update_gc, lr, offline=False, gc_only=False, intr_coef=0,switch_gc=500000, load_protos=False,num_protos=512):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    cfg.goal_shape = goal_shape
    cfg.goal = goal
    cfg.hidden_dim = hidden_dim
    cfg.batch_size = batch_size
    cfg.update_gc = update_gc
    cfg.lr = lr
    cfg.offline = offline
    cfg.gc_only = gc_only
    if self.cfg.film_gc:
        cfg.switch_gc = switch_gc
    if cfg.name=='proto_intr':
        cfg.intr_coef = intr_coef
    cfg.load_protos = load_protos
    cfg.num_protos=num_protos
    return hydra.utils.instantiate(cfg)

def get_state_embeddings(agent, states):
    with torch.no_grad():
        s = agent.encoder(states)
        s = agent.predictor(s)
        s = agent.projector(s)
        s = F.normalize(s, dim=1, p=2)
    return s

def encoding_grid(agent, work_dir, cfg, env, model_step):
    replay_dir = work_dir / 'buffer2' / 'buffer_copy'
    print('make encoding grid buffer')
    replay_buffer = make_replay_offline(env,
                                        replay_dir,
                                        100000,
                                        cfg.batch_size,
                                        0,
                                        cfg.discount,
                                        goal=False,
                                        relabel=False,
                                        model_step = model_step,
                                        replay_dir2=False,
                                        obs_type = cfg.obs_type
                                        )
    pix, states, actions = replay_buffer._sample(eval_pixel=True)
    if states == '':
        print('nothing in buffer yet')
    else:
        pix = pix.astype(np.float64)
        states = states.astype(np.float64)
        states = states.reshape(-1,2)
        grid = pix.reshape(-1,9, 84, 84)
        grid = torch.tensor(grid).cuda().float()
        grid = get_state_embeddings(agent, grid)
        return grid, states

    
def heatmaps(self, env, model_step, replay_dir2, goal,model_step_lb=False,gc=False,proto=False):
        if gc:

            heatmap = self.replay_storage1.state_visitation_gc

            plt.clf()
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
            ax.set_title(model_step)

            plt.savefig(f"./{model_step}_gc_heatmap.png")
            wandb.save(f"./{model_step}_gc_heatmap.png")


            heatmap_pct = self.replay_storage1.state_visitation_gc_pct

            plt.clf()
            fig, ax = plt.subplots(figsize=(10,10))
            labels = np.round(heatmap_pct.T/heatmap_pct.sum()*100, 1)
            sns.heatmap(np.log(1 + heatmap_pct.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
            ax.set_title(model_step)

            plt.savefig(f"./{model_step}_gc_heatmap_pct.png")
            wandb.save(f"./{model_step}_gc_heatmap_pct.png")


            reward_matrix = self.replay_storage1.reward_matrix
            plt.clf()
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(np.log(1 + reward_matrix.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
            ax.set_title(model_step)

            plt.savefig(f"./{model_step}_gc_reward.png")
            wandb.save(f"./{model_step}_gc_reward.png")
            
            goal_matrix = self.replay_storage1.goal_state_matrix
            plt.clf()
            fig, ax = plt.subplots(figsize=(10,10))
            labels = np.round(goal_matrix.T/goal_matrix.sum()*100, 1)
            sns.heatmap(np.log(1 + goal_matrix.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
            ax.set_title(model_step)

            plt.savefig(f"./{model_step}_goal_state_heatmap.png")
            wandb.save(f"./{model_step}_goal_state_heatmap.png")
        if proto:

            heatmap = self.replay_storage2.state_visitation_proto

            plt.clf()
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
            ax.set_title(model_step)

            plt.savefig(f"./{model_step}_proto_heatmap.png")
            wandb.save(f"./{model_step}_proto_heatmap.png")


            heatmap_pct = self.replay_storage2.state_visitation_proto_pct

            plt.clf()
            fig, ax = plt.subplots(figsize=(10,10))
            labels = np.round(heatmap_pct.T/heatmap_pct.sum()*100, 1)
            sns.heatmap(np.log(1 + heatmap_pct.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
            ax.set_title(model_step)

            plt.savefig(f"./{model_step}_proto_heatmap_pct.png")
            wandb.save(f"./{model_step}_proto_heatmap_pct.png")

        
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
        #task = PRIMAL_TASKS[self.cfg.domain]
        self.no_goal_task = 'point_mass_maze_reach_no_goal'
        idx = np.random.randint(0,400)
        goal_array = ndim_grid(2,20)
        self.first_goal = np.array([goal_array[idx][0], goal_array[idx][1]])
        self.train_env1 = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                   cfg.action_repeat, seed=None, goal=self.first_goal)
        print('goal', self.first_goal)
        self.train_env_no_goal = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                   cfg.action_repeat, seed=None, goal=None)
        #import IPython as ipy; ipy.embed(colors='neutral')
        print('no goal task env', self.no_goal_task)
        self.train_env_goal = dmc.make(self.no_goal_task, 'states', cfg.frame_stack,
                                   1, seed=None, goal=None)
        self.eval_env = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, seed=None, goal=self.first_goal)
        self.eval_env_no_goal = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                   cfg.action_repeat, seed=None, goal=None)
        self.eval_env_goal = dmc.make(self.no_goal_task, 'states', cfg.frame_stack,
                                   1, seed=None, goal=None)
        self.goal_queue = np.zeros((50, 2))
        self.goal_queue_ptr = 0 
        self.goal_array = ndim_grid(2,20)
        lst =[]
        for ix,x in enumerate(self.goal_array):
            print(x[0])
            print(x[1])
            if (-.2<x[0]<.2 and -.02<x[1]<.02) or (-.02<x[0]<.02 and -.2<x[1]<.2):
                lst.append(ix)
                print('del',x)
        self.goal_array=np.delete(self.goal_array, lst,0)
        self.curriculum_goal_loaded=False
        self.fully_loaded=False 
        # create agent
        #import IPython as ipy; ipy.embed(colors='neutral')
        if self.cfg.film_gc:
            self.agent = make_agent(self,
                                cfg.obs_type,
                                self.train_env1.observation_spec(),
                                self.train_env1.action_spec(),
                                (9, 84, 84),
                                cfg.num_seed_frames // cfg.action_repeat,
                                True,
                                cfg.agent,
                                cfg.hidden_dim,
                                cfg.batch_size,
                                cfg.update_gc,
                                cfg.lr,
                                cfg.offline,
                                gc_only=True,
                                intr_coef=cfg.intr_coef,
                                switch_gc=cfg.switch_gc,
                                )
        else:
            self.agent = make_agent(self,
                                cfg.obs_type,
                                self.train_env1.observation_spec(),
                                self.train_env1.action_spec(),
                                (9, 84, 84),
                                cfg.num_seed_frames // cfg.action_repeat,
                                True,
                                cfg.agent,
                                cfg.hidden_dim,
                                cfg.batch_size,
                                cfg.update_gc,
                                cfg.lr,
                                cfg.offline,
                                gc_only=True,
                                intr_coef=cfg.intr_coef,
                                load_protos=False,
                                num_protos=cfg.num_protos) 
        
        if self.cfg.load_encoder and self.cfg.load_proto==False:

            encoder = torch.load('/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/encoder/2022.09.09/072830_proto_lambda/encoder_proto_1000000.pth')
            #encoder = torch.load('/home/ubuntu/proto_explore/url_benchmark/exp_local/2022.09.09/072830_proto/encoder_proto_1000000.pth')
            #encoder = torch.load('/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/models/encoder/2022.09.09/072830_proto_lambda/encoder_proto_1000000.pth')
            self.agent.init_encoder_from(encoder)
        if self.cfg.load_proto:
            #proto  = torch.load('/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/models/encoder/2022.09.09/072830_proto_lambda/optimizer_proto_1000000.pth')
            proto = torch.load('/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/encoder/2022.09.09/072830_proto_lambda/optimizer_proto_1000000.pth')
            #proto  = torch.load('/home/ubuntu/proto_explore/url_benchmark/exp_local/2022.09.09/072830_proto/optimizer_proto_1000000.pth')
            #proto  = torch.load('/home/ubuntu/proto_explore/url_benchmark/exp_local/2022.09.09/072830_proto/optimizer_proto_1000000.pth')
            self.agent.init_protos_from(proto) 

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env1.observation_spec(),
                      self.train_env1.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage1 = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer1')
      #  self.replay_storage2 = ReplayBufferStorage(data_specs, meta_specs,
      #                                            self.work_dir / 'buffer2')
        self.replay_goal_dir = Path('/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/exp_local/2022.09.07/144129_proto/buffer2/buffer_copy/') 

        # create replay buffer
        if cfg.offline:
            #might have conflict 
            self.replay_loader1 = make_replay_loader(self.replay_storage2,
                                                    False,
                                                    False,
                                                    cfg.replay_buffer_size,
                                                    cfg.batch_size_gc,
                                                    cfg.replay_buffer_num_workers,
                                                    False,
                                                    cfg.nstep,
                                                    cfg.discount,
                                                    True,
                                                    False,
                                                    cfg.obs_type,
                                                    0)
        elif cfg.offline_online or cfg.hybrid:
            print('making it later')
        else:
            print('regular or hybrid_gc loader')
            self.replay_iterable, self.replay_loader1 = make_replay_loader(self.replay_storage1,
                                                    False,
                                                    cfg.replay_buffer_gc,
                                                    cfg.batch_size_gc,
                                                    cfg.replay_buffer_num_workers,
                                                    False, cfg.nstep, cfg.discount,
                                                    True, cfg.hybrid_gc,cfg.obs_type,
                                                    cfg.hybrid_pct,return_iterable=True)
             

        self._replay_iter1 = None


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
        self.unreachable_goal = np.empty((0,9,84,84))
        self.unreachable_state = np.empty((0,2))
        self.loaded = False
        self.loaded_uniform = False
        self.uniform_goal = []
        self.uniform_state = []
        self.count_uniform = 0 
        self.goal_loaded = False
        self.distance_goal = []
        self.count=0
        self.global_success_rate = []
        self.global_index=[]
        self.storage1=False
        self.proto_goal = []
        self.distance_goal_dict={} 
        self.resampled=False
        self.reload_goal=False
    
    
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

    
    
    #def sample_goal_proto(self, obs):
    #    #current_protos = self.agent.protos.weight.data.clone()
    #    #current_protos = F.normalize(current_protos, dim=1, p=2)
    #    if len(self.unreachable) > 0:
    #        print('list of unreachables', self.unreachable)
    #        return self.unreachable.pop(0)
    #    else:
    #        proto2d = #sample prototypes 
    #        num = proto2d.shape[0]
    #        idx = np.random.randint(0, num)
    #        return proto2d[idx,:].cpu().numpy()
    
#     def encoding_grid():
#         if self.loaded == False:
#             replay_dir = self.work_dir / 'buffer2' / 'buffer_copy'
#             print('make encoding grid buffer 2')
#             self.replay_buffer_intr = make_replay_offline(self.eval_env,
#                                     replay_dir,
#                                     100000,
#                                     self.cfg.batch_size,
#                                     0,
#                                     self.cfg.discount,
#                                     goal=False,
#                                     relabel=False,
#                                     model_step = self.global_step,
#                                     replay_dir2=False,
#                                     obs_type = self.cfg.obs_type
#                                     )
#             self.loaded = True
#             pix, states, actions = self.replay_buffer_intr._sample(eval_pixel=True)
#             pix = pix.astype(np.float64)
#             states = states.astype(np.float64)
#             states = states.reshape(-1,2)
#             grid = pix.reshape(-1,9, 84, 84)
#             grid = torch.tensor(grid).cuda().float()
#             return grid, states
#         else:
#             pix, states, actions = self.replay_buffer_intr._sample(eval_pixel=True)
#             pix = pix.astype(np.float64)
#             states = states.astype(np.float64)
#             states = states.reshape(-1,2)
#             grid = pix.reshape(-1,9, 84, 84)
#             grid = torch.tensor(grid).cuda().float()
#             return grid, states

    def encode_proto(self, heatmap_only=False):
        if self.cfg.film_gc:
            replay_dir = self.work_dir / 'buffer1' / 'buffer_copy'
            self.replay_buffer_intr = make_replay_offline(self.eval_env,
                                    replay_dir,
                                    100000,
                                    self.cfg.batch_size,
                                    0,
                                    self.cfg.discount,
                                    goal=False,
                                    relabel=False,
                                    model_step = self.global_step,
                                    replay_dir2=False,
                                    obs_type = self.cfg.obs_type
                                    )
        elif self.cfg.load_proto and heatmap_only==False:
            replay_dir = '/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/models/encoder/2022.09.09/072830_proto_lambda/buffer2/buffer_copy/'
            self.replay_buffer_intr = make_replay_offline(self.eval_env,
                                        Path(replay_dir),
                                        100000,
                                        self.cfg.batch_size,
                                        0,
                                        self.cfg.discount,
                                        goal=False,
                                        relabel=False,
                                        model_step =max(3000,self.global_step),
                                        replay_dir2=False,
                                        obs_type = self.cfg.obs_type
                                        )
        else:
            
            replay_dir = self.work_dir / 'buffer1' / 'buffer_copy'
            self.replay_buffer_intr = make_replay_offline(self.eval_env,
                                        replay_dir,
                                        100000,
                                        self.cfg.batch_size,
                                        0,
                                        self.cfg.discount,
                                        goal=False,
                                        relabel=False,
                                        model_step = self.global_step,
                                        replay_dir2=False,
                                        obs_type = self.cfg.obs_type
                                        )
            
        
        if heatmap_only:
            states, actions, rewards, goal_state = self.replay_buffer_intr.parse_dataset(goal_state=True)
            tmp = np.hstack((states, goal_state, rewards))
            df = pd.DataFrame(tmp, columns= ['x', 'y', 'pos', 'v', 'g_x', 'g_y', 'gp', 'gv','r'])
            heatmap, _, _ = np.histogram2d(df.iloc[:, 0], df.iloc[:, 1], bins=50,
                                   range=np.array(([-.29, .29],[-.29, .29])))
            plt.clf()
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
            ax.set_title(self.global_step)
            plt.savefig(f"./{self.global_step}_gc_heatmap.png")
            wandb.save(f"./{self.global_step}_gc_heatmap.png")

            heatmap, _, _ = np.histogram2d(df.iloc[:, 4], df.iloc[:, 5], bins=50,
                                   range=np.array(([-.29, .29],[-.29, .29])))
            plt.clf()
            fig, ax = plt.subplots(figsize=(10,6))
            sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
            ax.set_title(self.global_step)
            plt.savefig(f"./{self.global_step}_gc_goals.png")
            wandb.save(f"./{self.global_step}_gc_goals.png")

            #percentage breakdown
            df=df*100
            heatmap, _, _ = np.histogram2d(df.iloc[:, 0], df.iloc[:, 1], bins=20,
                                   range=np.array(([-29, 29],[-29, 29])))
            plt.clf()

            fig, ax = plt.subplots(figsize=(10,10))
            labels = np.round(heatmap.T/heatmap.sum()*100, 1)
            sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax, annot=labels).invert_yaxis()
            plt.savefig(f"./{self._global_step}_gc_heatmap_pct.png")
            wandb.save(f"./{self._global_step}_gc_heatmap_pct.png")
    
            #rewards seen thus far
            df = df.astype(int)
            result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['r']
            result.fillna(0, inplace=True)
            plt.clf()
            fig, ax = plt.subplots()
            sns.heatmap(result, cmap="Blues_r", ax=ax).invert_yaxis()
            plt.savefig(f"./{self._global_step}_gc_reward.png")
            wandb.save(f"./{self._global_step}_gc_reward.png")

        if heatmap_only==False:
            state, actions, rewards = self.replay_buffer_intr.parse_dataset()
            idx = np.random.randint(0, state.shape[0], size=1024)
            state=state[idx]
            state=state.reshape(1024,4)
            print('state shape',state.shape)
            obs = torch.empty(1024, 9, 84, 84)
            states = np.empty((1024,4),np.float)
            grid_embeddings = torch.empty(1024, 128)
            meta = self.agent.init_meta()
            for i in range(1024):
                with torch.no_grad():
                    with self.eval_env_goal.physics.reset_context():
                        time_step = self.eval_env_goal.physics.set_state(np.array([state[i][0], state[i][1], 0, 0]))
                    time_step = self.eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
                    goal = torch.tensor(np.transpose(time_step.copy(), (2,0,1))).cuda()
                    obs[i] = torch.tile(goal, (1,3,1,1))
                    states[i] = state[i,:]
            obs = obs.cuda().float()
            grid_embeddings = get_state_embeddings(self.agent, obs)
            protos = self.agent.protos.weight.data.detach().clone()
            protos = F.normalize(protos, dim=1, p=2)
            dist_mat = torch.cdist(protos, grid_embeddings)
            dist_np = dist_mat.cpu().numpy()
            dist_df = pd.DataFrame(dist_np)	
            dist_df.to_csv('./dist_{}.csv'.format(self.global_step), index=False)
            closest_points = dist_mat.argmin(-1)
            proto2d = states[closest_points.cpu(), :2]
            states = pd.DataFrame(states)
            states.to_csv('./states_{}.csv'.format(self.global_step), index=False)
            return proto2d
    

    def sample_goal_distance(self,init_state_idx=None,init_state=False):

        if self.goal_loaded==False:

            goal_array = ndim_grid(2,20)
            
            if init_state_idx is None:
                if init_state==False:
                    dist_goal = cdist(np.array([[-.15,.15]]), goal_array, 'euclidean')  
                else:
                    dist_goal = cdist(np.array([[init_state[0],init_state[1]]]), goal_array, 'euclidean')
                
                df1=pd.DataFrame()
                df1['distance'] = dist_goal.reshape((400,))
                df1['index'] = df1.index
                df1 = df1.sort_values(by='distance')
                goal_array_ = []
                for x in range(len(df1)):
                    goal_array_.append(goal_array[df1.iloc[x,1]])
                self.distance_goal = goal_array_
                self.goal_loaded=True
                index=self.global_step//1000
                idx = np.random.randint(index,min(index+30, 400))
            else:
                dist_goal0 = cdist(np.array([[.15,.15]]), goal_array, 'euclidean')
                dist_goal1 = cdist(np.array([[.15,-.15]]), goal_array, 'euclidean')
                dist_goal2 = cdist(np.array([[-.15,.15]]), goal_array, 'euclidean')
                dist_goal3 = cdist(np.array([[-.15,-.15]]), goal_array, 'euclidean')

                for ix,i in enumerate([dist_goal0,dist_goal1,dist_goal2,dist_goal3]):
                    df1=pd.DataFrame()
                    df1['distance'] = i.reshape((400,))
                    df1['index'] = df1.index
                    df1 = df1.sort_values(by='distance')
                    goal_array_ = []
                    for x in range(len(df1)):
                        goal_array_.append(goal_array[df1.iloc[x,1]])
                    self.distance_goal_dict[ix] = goal_array_
                self.goal_loaded=True
                index=self.global_step//2000
                idx = np.random.randint(max(index-10,0),min(index+20, 400))


        else:
            if self.global_step<500000:
                index=self.global_step//2000
                idx = np.random.randint(max(index-10,0),min(index+20, 400))
            else:
                index=(self.global_step-500000)//2000
                idx = np.random.randint(max(index-10,0),min(index+20, 400)) 
        
        if init_state_idx is None:
            return self.distance_goal[idx]
        else:
            return self.distance_goal_dict[init_state_idx][idx]
    
    def sample_goal_pixel(self, eval=False):
        replay_dir = self.work_dir / "buffer2" / "buffer_copy"
    #    if len(self.unreachable_goal) > 0 and eval==False:
    #        a = [tuple(row) for row in self.unreachable_state]
    #        idx = np.unique(a, axis=0, return_index=True)[1]
    #        self.unreachable_state = self.unreachable_state[idx]
    #        self.unreachable_goal = self.unreachable_goal[idx]
    #        print('list of unreachables', self.unreachable_state)
    #        obs = self.unreachable_goal[0]
    #        state = self.unreachable_state[0]
    #        self.unreachable_state = np.delete(self.unreachable_state, 0, 0)
    #        self.unreachable_goal = np.delete(self.unreachable_goal, 0, 0)
    #        return obs, state

        if (self.global_step<100000 and self.global_step%20000==0 and eval==False) or (self.global_step %100000==0 and eval==False):
            self.replay_buffer_goal = make_replay_buffer(self.eval_env,
                                                        replay_dir,
                                                        50000,
                                                        self.cfg.batch_size,
                                                        0,
                                                        self.cfg.discount,
                                                        goal=False,
                                                        relabel=False,
                                                        replay_dir2 = False,
                                                        obs_type=self.cfg.obs_type,
                                                        model_step=self.global_step                                                                                                          )
            obs, state = self.replay_buffer_goal._sample_pixel_goal(self.global_step)
            return obs, state
        else:
            obs, state = self.replay_buffer_goal._sample_pixel_goal(self.global_step)
            return obs, state
    

    def eval_intrinsic(self, model):
        grid_embeddings = torch.empty(1024, 9, 84, 84)
        states = torch.empty(1024, 2)
        for i in range(1024):
            grid, state = encoding_grid(self.agent, self.work_dir, self.cfg, self.eval_env, model)
            grid_embeddings[i] = grid
            states[i] = torch.tensor(state).cuda().float()
        protos = self.agent.protos.weight.data.detach().clone()
        protos = F.normalize(protos, dim=1, p=2)
        dist_mat = torch.cdist(protos, grid_embeddings)
        closest_points = dist_mat.argmin(-1)
        proto2d = states[closest_points.cpu(), :2]

        meta = self.agent.init_meta() 
        with torch.no_grad():
            reward = self.agent.compute_intr_reward(grid_embeddings, self._global_step)
            action = self.agent.act2(obs, meta, self._global_step, eval_mode=True)
            q = self.agent.get_q_value(obs, action)
        for x in range(len(reward)):
            print('saving')
            print(str(self.work_dir)+'/eval_intr_reward_{}.csv'.format(self._global_step))
            save(str(self.work_dir)+'/eval_intr_reward_{}.csv'.format(self._global_step), [[obs[x].cpu().detach().numpy(), reward[x].cpu().detach().numpy(), q[x].cpu().detach().numpy(), self._global_step]])

    def eval(self):
        #self.encode_proto(heatmap_only=True) 
        heatmaps(self, self.eval_env, self.global_step, False, True, model_step_lb=False,gc=True,proto=False)
        goal_array = ndim_grid(2,10)
        success=0
        df = pd.DataFrame(columns=['x','y','r'], dtype=np.float64) 

        for ix, x in enumerate(goal_array):
            step, episode, total_reward = 0, 0, 0
         #   goal_pix, goal_state = self.sample_goal_uniform(eval=True)
            goal_state = np.array([x[0], x[1]])
            self.eval_env = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                    self.cfg.action_repeat, seed=None, goal=goal_state)
            self.eval_env_no_goal = dmc.make(self.no_goal_task, self.cfg.obs_type, self.cfg.frame_stack,
                    self.cfg.action_repeat, seed=None, goal=None)
            self.eval_env_goal = dmc.make(self.no_goal_task, 'states', self.cfg.frame_stack,
                    self.cfg.action_repeat, seed=None, goal=None)
            eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
            meta = self.agent.init_meta()

            while eval_until_episode(episode):
                time_step = self.eval_env.reset()
                time_step_no_goal = self.eval_env_no_goal.reset()

                with self.eval_env_goal.physics.reset_context():
                    time_step_goal = self.eval_env_goal.physics.set_state(np.array([goal_state[0], goal_state[1], 0, 0]))
                time_step_goal = self.eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
                self.video_recorder.init(self.eval_env, enabled=(episode == 0))
         
                while step!=self.cfg.episode_length:
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        if self.cfg.goal:
                            action = self.agent.act(time_step_no_goal.observation['pixels'],
                                                    time_step_goal,
                                                    meta,
                                                    self._global_step,
                                                    eval_mode=True)
                        else:
                            action = self.agent.act(time_step.observation,
                                                    meta,
                                                    self._global_step,
                                                    eval_mode=True)
                    time_step = self.eval_env.step(action)
                    time_step_no_goal = self.eval_env_no_goal.step(action)
                    #time_step_goal = self.eval_env_goal.step(action)
                    self.video_recorder.record(self.eval_env)
                    total_reward += time_step.reward
                    step += 1

                episode += 1
                self.video_recorder.save(f'{self.global_frame}_{ix}.mp4')
                
                if ix%10==0:
                    wandb.save(f'{self.global_frame}_{ix}.mp4')

                if self.cfg.eval:
                    print('saving')
                    save(str(self.work_dir)+'/eval_{}.csv'.format(model.split('.')[-2].split('/')[-1]), [[x.cpu().detach().numpy(), total_reward, time_step.observation[:2], step]])

                else:
                    print('saving')
                    print(str(self.work_dir)+'/eval_{}.csv'.format(self._global_step))
                    save(str(self.work_dir)+'/eval_{}.csv'.format(self._global_step), [[goal_state, total_reward, time_step.observation['observations'], step]])
            
            if total_reward > 200*self.cfg.num_eval_episodes:
                success+=1
            
            df.loc[ix, 'x'] = x[0]
            df.loc[ix, 'y'] = x[1]
            df.loc[ix, 'r'] = total_reward

        result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['r']/2
        plt.clf()
        fig, ax = plt.subplots()
        sns.heatmap(result, cmap="Blues_r").invert_yaxis()
        plt.savefig(f"./{self.global_step}_heatmap.png")
        wandb.save(f"./{self.global_step}_heatmap.png")
        success_rate = success/len(goal_array)
        self.global_success_rate.append(success_rate)
        self.global_index.append(self.global_step)
        print('success_rate of current eval', success_rate)
        
        
        
        


 
    def train(self):
        # predicates
        resample_goal_every = 500
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step1 = self.train_env1.reset()
        self.train_env_no_goal = dmc.make(self.no_goal_task, self.cfg.obs_type, self.cfg.frame_stack,
                self.cfg.action_repeat, seed=None, goal=self.first_goal, init_state=time_step1.observation['observations'][:2])
        time_step_no_goal = self.train_env_no_goal.reset()
        time_step_goal = self.train_env_goal.reset()
        with self.train_env_goal.physics.reset_context():
            time_step_goal = self.train_env_goal.physics.set_state(np.array([self.first_goal[0], self.first_goal[1], 0, 0]))
        
        time_step_goal = self.train_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))

        meta = self.agent.init_meta() 
         
        if self.cfg.obs_type == 'pixels':
            self.replay_storage1.add_goal(time_step1, meta, time_step_goal, time_step_no_goal,self.train_env_goal.physics.state(), True)
            print('replay1')

        else:
            self.replay_storage1.add_goal(time_step1, meta, goal)


        #self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            #self.train_video_recorder.init(self.train_env_goal, enabled=(episode_step == 0))

            if self.cfg.offline:
                if eval_every_step(self.global_step) and self.global_step!=0:
                    self.eval()

                metrics = self.agent.update(self.replay_iter1, self.global_step, actor1=True)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
                
                if self.global_step%self.cfg.episode_length==0:
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                          ty='train') as log:

                        log('fps', 1000 / elapsed_time)
                        log("total_time", total_time)
                        log("step", self.global_step)
                self._global_step += 1
                
            else:
                
                if time_step1.last() or episode_step==self.cfg.episode_length:
                    print('last')
                    self._global_episode += 1
                    #self.train_video_recorder.save(f'{self.global_frame}.mp4')
                    # wait until all the metrics schema is populated
                    if metrics is not None:
                        # log stats
                        elapsed_time, total_time = self.timer.reset()
                        episode_frame = episode_step * self.cfg.action_repeat
                        with self.logger.log_and_dump_ctx(self.global_frame,ty='train') as log:
                            log('fps', episode_frame / elapsed_time)
                            log('total_time', total_time)
                            log('episode_reward', episode_reward)
                            log('episode_length', episode_frame)
                            log('episode', self.global_episode)
                            log('buffer_size', len(self.replay_storage1))
                            log('step', self.global_step)


                    if self.cfg.obs_type =='pixels':
                        self.replay_storage1.add_goal(time_step1, meta,time_step_goal, time_step_no_goal, self.train_env_goal.physics.state(), True, last=True)
                        
                    else:
                        self.replay_storage.add(time_step, meta)

                    #self.train_video_recorder.init(time_step.observation)
                    # try to save snapshot
                    if self.global_frame in self.cfg.snapshots:
                        self.save_snapshot()
                    episode_step = 0
                    episode_reward = 0
                    self.resampled=False

                # try to evaluate
                if eval_every_step(self.global_step) and self.global_step!=0:
                    #print('trying to evaluate')
                    self.eval()
                    plt.clf()
                    fig, ax = plt.subplots()
                    sr, idx = zip(*sorted(zip(self.global_success_rate, self.global_index)))

                    ax.ticklabel_format(style='plain')
                    ax.plot(idx,sr)
                    plt.savefig(f"./{self._global_step}_eval.png")
                    wandb.save("./{self._global_step}_eval.png")

                if (episode_step== 0 and self.global_step!=0) or (episode_reward<50 and episode_step==250 and self.global_step!=0 and self.resampled==False):
                    if self.cfg.curriculum:
                        print('sampling goal')
                    elif self.cfg.sample_proto:
                        if self.cfg.load_proto==False:
                            if (self.cfg.num_seed_frames==self.global_step) or (self.global_step < 100000 and self.global_step%2000==0) or (300000>self.global_step >100000 and self.global_step%50000==0):
                                if self.global_step%100000==0:
                                    self.proto_goal = self.encode_proto()
                                else:
                                    self.proto_goal = self.encode_proto()
                                dist_goal = cdist(np.array([[-.15,.15]]), np.array(self.proto_goal), 'euclidean')
                                df1 = pd.DataFrame()
                                df1['distance'] = dist_goal.reshape((len(self.proto_goal),))
                                df1['index'] = df1.index
                                df1 = df1.sort_values(by='distance',ascending=False)
                                goal_array_ = []
                                for x in range(len(df1)):
                                    goal_array_.append(self.proto_goal[df1.iloc[x,1]])
                                self.proto_goal = goal_array_
                                idx = np.random.randint(0, len(self.proto_goal))
                                goal_state = np.array([self.proto_goal[idx][0], self.proto_goal[idx][1]])

                            elif len(self.proto_goal)>0:

                                idx = min(int(np.random.exponential(max(int(len(self.proto_goal)/5),1))), len(self.proto_goal)-1)
                                goal_state = np.array([self.proto_goal[idx][0], self.proto_goal[idx][1]])

                            else:
                                print('havent sampled prototypes yet, sampling randomly')
                                goal_array = ndim_grid(2,20)
                                idx = np.random.randint(0,len(goal_array))
                                goal_state = np.array([goal_array[idx][0], goal_array[idx][1]])
                        #if self.cfg.load_proto
                        else:
                            if self.loaded==False:
                                self.proto_goal = self.encode_proto() 
                                self.loaded=True
                                dist_goal = cdist(np.array([[-.15,.15]]), np.array(self.proto_goal), 'euclidean')
                                df1 = pd.DataFrame()
                                df1['distance'] = dist_goal.reshape((len(self.proto_goal),))
                                df1['index'] = df1.index
                                df1 = df1.sort_values(by='distance',ascending=False)
                                goal_array_ = []
                                for x in range(len(df1)):
                                    goal_array_.append(self.proto_goal[df1.iloc[x,1]])
                                self.proto_goal = goal_array_
                                idx = min(int(np.random.exponential(max(int(len(self.proto_goal)/5),1))), len(self.proto_goal)-1)
                                goal_state = np.array([self.proto_goal[idx][0], self.proto_goal[idx][1]])
                            else:
                                if self.global_step%50000==0:
                                    self.loaded=False
                                dist_goal = cdist(np.array([[-.15,.15]]), np.array(self.proto_goal), 'euclidean')
                                #dist_goal = cdist(np.array([self.proto_goal]).reshape(-1,1), np.array([[-.15,.15,0,0]]),'euclidean')

                                df1 = pd.DataFrame()
                                df1['distance'] = dist_goal.reshape((len(self.proto_goal),))
                                df1['index'] = df1.index
                                df1 = df1.sort_values(by='distance',ascending=False)
                                goal_array_ = []
                                for x in range(len(df1)):
                                    goal_array_.append(self.proto_goal[df1.iloc[x,1]])
                                self.proto_goal = goal_array_
                                idx = min(int(np.random.exponential(max(int(len(self.proto_goal)/5),1))), len(self.proto_goal)-1)
                                goal_state = np.array([self.proto_goal[idx][0], self.proto_goal[idx][1]])

                    else:
                        
                        goal_array = ndim_grid(2,20)
                        idx = self.count
                        print('count', self.count)
                        self.count += 1
                        if self.count == len(goal_array):
                            self.count = 0
                        goal_state = np.array([goal_array[idx][0], goal_array[idx][1]])
                    
                    if self.cfg.const_init==False:
                        initiation = np.array([[1,1],[1,-1],[-1,1],[-1,-1]])
                        initial = np.array([.15, .15])
                        init_rand = np.random.randint(4)
                        init_state = np.array([initial[0]*initiation[init_rand][0], initial[1]*initiation[init_rand][1]])
                    
                    if self.cfg.curriculum:
                        if self.curriculum_goal_loaded==False:
                            if self.cfg.const_init==False:
                                goal_=self.sample_goal_distance(init_rand)
                            else:
                                goal_=self.sample_goal_distance()
                            goal_state = np.array([goal_[0], goal_[1]]) 
                        
                        else:
                            print('goals left to reach', self.goal_array.shape[0])
                            if self.goal_array.shape[0]>10:
                                if self.global_step%5000==0:
                                    ix = np.random.uniform(.02,.29,(2,))
                                    sign = np.array([[1,1],[1,-1],[-1,-1]])
                                    rand = np.random.randint(3)
                                    goal_state = np.array([ix[0]*sign[rand][0], ix[1]*sign[rand][1]])
                                else:
                                    idx = np.random.randint(self.goal_queue.shape[0])
                                    goal_state = self.goal_queue[idx]
                            else:
                                
                                if self.reload_goal ==False:
                                    self.replay_iterable.hybrid=False
                                    self.reload_goal_array = ndim_grid(2,20)
                                    lst =[]
                                    for ix,x in enumerate(self.reload_goal_array):
                                        print(x[0])
                                        print(x[1])
                                        if (-.2<x[0]<.2 and -.02<x[1]<.02) or (-.02<x[0]<.02 and -.2<x[1]<.2):
                                            lst.append(ix)
                                            print('del',x)
                                    self.reload_goal_array=np.delete(self.reload_goal_array, lst,0)
                                    idx = np.random.randint(self.reload_goal_array.shape[0])
                                    goal_state= self.reload_goal_array[idx]
                                    self.reload_goal=True
                                else:
                                    idx = np.random.randint(self.reload_goal_array.shape[0])
                                    goal_state= self.reload_goal_array[idx]
                                
                                

                    if self.cfg.const_init==False and episode_step==0:
                        self.train_env1 = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                                                      self.cfg.action_repeat, seed=None, goal=goal_state,init_state=init_state)
                    elif self.cfg.const_init and episode_step==0:
                        self.train_env1 = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                                                      self.cfg.action_repeat, seed=None, goal=goal_state)
                    elif episode_step==250:

                        print('no reward for 250')
                        current_state = time_step1.observation['observations']
                        dist_goal = cdist(np.array([[current_state[0],current_state[1]]]), self.goal_array, 'euclidean')

                        df1=pd.DataFrame()
                        df1['distance'] = dist_goal.reshape((dist_goal.shape[1],))
                        df1['index'] = df1.index
                        df1 = df1.sort_values(by='distance')
                        goal_array_ = []
                        for x in range(len(df1)):
                            goal_array_.append(self.goal_array[df1.iloc[x,1]]) 
                        goal_state = goal_array_[1]
                        self.train_env1 = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                                                      self.cfg.action_repeat, seed=None, goal=goal_state,init_state=np.array([current_state[0], current_state[1]]))
                    
                    time_step1 = self.train_env1.reset()
                    self.train_env_no_goal = dmc.make(self.no_goal_task, self.cfg.obs_type, self.cfg.frame_stack,
                    self.cfg.action_repeat, seed=None, goal=goal_state, init_state=time_step1.observation['observations'][:2])
                    time_step_no_goal = self.train_env_no_goal.reset()
                    meta = self.agent.update_meta(meta, self._global_step, time_step1) 
                    print('sampled goal', goal_state)

                    with self.train_env_goal.physics.reset_context():
                        time_step_goal = self.train_env_goal.physics.set_state(np.array([goal_state[0], goal_state[1],0,0]))

                    time_step_goal = self.train_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))

                    if self.cfg.obs_type == 'pixels' and time_step1.last()==False:
                        self.replay_storage1.add_goal(time_step1, meta,time_step_goal, time_step_no_goal,self.train_env_goal.physics.state(), True)

                # sample action
                with torch.no_grad(), utils.eval_mode(self.agent):
                    if self.cfg.obs_type == 'pixels':

                        action1 = self.agent.act(time_step_no_goal.observation['pixels'].copy(),
                                                time_step_goal.copy(),
                                                meta,
                                                self._global_step,
                                                eval_mode=False)
                    else:
                        action = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=False)

                # take env step
                time_step1 = self.train_env1.step(action1)
                time_step_no_goal = self.train_env_no_goal.step(action1)
                episode_reward += time_step1.reward

                if self.cfg.obs_type == 'pixels' and time_step1.last()==False:
                    self.replay_storage1.add_goal(time_step1, meta, time_step_goal, time_step_no_goal,self.train_env_goal.physics.state(), True)
                elif self.cfg.obs_type == 'states':
                    self.replay_storage1.add_goal(time_step1, meta, goal)

                episode_step += 1
                
                if episode_reward > 100 and self.cfg.resample_goal and self.reload_goal==False:
                    print('reached making new env')
                    self.resampled=True
                    episode_reward=0
                    if goal_state.tolist() in self.goal_array.tolist():
                        ix = self.goal_array.tolist().index(goal_state.tolist())
                        self.goal_array=np.delete(self.goal_array, ix, 0)
                        
                    print('goals left', self.goal_array.shape[0]) 
                    init_state = goal_state
                    
                    dist_goal = cdist(np.array([[init_state[0],init_state[1]]]), self.goal_array, 'euclidean')
                
                    df1=pd.DataFrame()
                    df1['distance'] = dist_goal.reshape((dist_goal.shape[1],))
                    df1['index'] = df1.index
                    df1 = df1.sort_values(by='distance')
                    goal_array_ = []
                    for x in range(len(df1)):
                        goal_array_.append(self.goal_array[df1.iloc[x,1]])

                    for x in range(5):
                        ptr = self.goal_queue_ptr
                        self.goal_queue[ptr] = goal_array_[x]
                        self.goal_queue_ptr = (ptr + 1) % self.goal_queue.shape[0]
                    
                    if self.goal_queue_ptr==0:
                        self.curriculum_goal_loaded=True
                    print('reached making new env')
                    episode_reward=0
                    current_state = time_step1.observation['observations'][:2]
                    print('current_state', current_state)
                            
                    if self.cfg.curriculum:
                        if self.goal_queue_ptr!=0:
                            idx = np.random.randint(self.goal_queue_ptr)
                        else:
                            idx = np.random.randint(self.goal_queue.shape[0])
                        goal_state = np.array([self.goal_queue[idx][0], self.goal_queue[idx][1]])
                    else:
                        idx = np.random.randint(0,400)
                        goal_array = ndim_grid(2,20)
                        goal_state = np.array([goal_array[idx][0], goal_array[idx][1]])

                    self.train_env1 = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                                                  self.cfg.action_repeat, seed=None, goal=goal_state, init_state=(current_state[0], current_state[1]))
                    print('should reset to', current_state)
                    print('new env state', self.train_env1._env.physics.state())
                    time_step1 = self.train_env1.reset()
                    print('reset state', time_step1.observation['observations'])
                    self.train_env_no_goal = dmc.make(self.no_goal_task, self.cfg.obs_type, self.cfg.frame_stack,
                                                  self.cfg.action_repeat, seed=None, goal=goal_state, init_state=(current_state[0], current_state[1]))
                    time_step_no_goal = self.train_env_no_goal.reset()
                    print('no goal reset', time_step_no_goal.observation['observations'])
                    #time_step_goal = self.train_env_goal.reset()
                    meta = self.agent.update_meta(meta, self._global_step, time_step1)
                    print('sampled goal', goal_state)

                    with self.train_env_goal.physics.reset_context():
                        time_step_goal = self.train_env_goal.physics.set_state(np.array([goal_state[0], goal_state[1],0,0]))

                    time_step_goal = self.train_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))

                    if self.cfg.obs_type == 'pixels' and time_step1.last()==False:
                        self.replay_storage1.add_goal(time_step1, meta,time_step_goal, time_step_no_goal,self.train_env_goal.physics.state(), True)

                if not seed_until_step(self.global_step):

                    if self.cfg.offline_online and self.storage1==False:
                        self.replay_loader1 = make_replay_buffer(self.eval_env,
                                                    self.work_dir/ "buffer2"/"buffer_copy",
                                                    self.cfg.replay_buffer_gc,
                                                    self.cfg.batch_size_gc,
                                                    self.cfg.replay_buffer_num_workers,
                                                    self.cfg.discount,
                                                    goal=True,
                                                    relabel=False,
                                                    replay_dir2=self.work_dir/"buffer1"/"buffer_copy",
                                                    obs_type=self.cfg.obs_type,
                                                    offline=True,
                                                    nstep=self.cfg.nstep)
                        self.storage1=True
                    elif self.cfg.hybrid and self.cfg.offline_online==False and self.storage1==False:
                        print('making hybrid buffer and offline_online=False')
                        self.replay_loader1 = make_replay_buffer(self.eval_env,
                                                    self.work_dir/ "buffer1",
                                                    self.cfg.replay_buffer_gc,
                                                    self.cfg.batch_size_gc,
                                                    self.cfg.replay_buffer_num_workers,
                                                    self.cfg.discount,
                                                    goal=True,
                                                    relabel=False,
                                                    replay_dir2=self.work_dir/"buffer2"/"buffer_copy",
                                                    obs_type=self.cfg.obs_type,
                                                    offline=False,
                                                    nstep=self.cfg.nstep,
                                                    hybrid=True,
                                                    hybrid_pct=self.cfg.hybrid_pct)
                        self.storage1=True

                    metrics = self.agent.update(self.replay_iter1, self.global_step, actor1=True)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

                self._global_step += 1

                
#                 if self._global_step == self.cfg.switch_gc and self.cfg.film_gc==True:
#                     print('updating film')
#                     #processors (adaptation networks) & regularization lists for each of 
#                     #the output params

#                     #trying residual instead of linear
#                     self.agent.film.gamma_1 = nn.Sequential(
#                         nn.Linear(self.cfg.feature_dim,self.cfg.feature_dim),
#                         nn.ReLU(),
#                         nn.Linear(self.cfg.feature_dim,self.cfg.feature_dim),
#                         nn.ReLU(),
#                         nn.Linear(self.cfg.feature_dim,self.cfg.feature_dim),
#                     )

#                     self.agent.film.gamma_1_regularizers = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(self.cfg.feature_dim), 0, 0.001),requires_grad=True)

#                     self.agent.film.gamma_2 = nn.Sequential(
#                         nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
#                         nn.ReLU(),
#                         nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
#                         nn.ReLU(),
#                         nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
#                     )

#                     self.agent.film.gamma_2_regularizers = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(self.cfg.hidden_dim), 0, 0.001),requires_grad=True)

#                     self.agent.film.beta_1 = nn.Sequential(
#                         nn.Linear(self.cfg.feature_dim,self.cfg.feature_dim),
#                         nn.ReLU(),
#                         nn.Linear(self.cfg.feature_dim,self.cfg.feature_dim),
#                         nn.ReLU(),
#                         nn.Linear(self.cfg.feature_dim,self.cfg.feature_dim),
#                     )

#                     self.agent.film.beta_1_regularizers = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(self.cfg.feature_dim), 0, 0.001),requires_grad=True)

#                     self.agent.film.beta_2 = nn.Sequential(
#                         nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
#                         nn.ReLU(),
#                         nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
#                         nn.ReLU(),
#                         nn.Linear(self.cfg.hidden_dim, self.cfg.hidden_dim),
#                         )

#                     self.agent.film.beta_2_regularizers = torch.nn.Parameter(torch.nn.init.normal_(torch.empty(self.cfg.hidden_dim), 0, 0.001),requires_grad=True)
                    
#                     self.agent.film.apply(utils.weight_init)
#                     #add in dense risdual layer's gradient

#                     self.agent.film_opt = torch.optim.Adam(self.agent.film.parameters(), lr=self.cfg.lr) 
#                     self.agent.film.train()
            
            if self._global_step%50000==0 and self._global_step!=0:
                print('saving agent')
                if self.cfg.gcsl==False:
                    path = os.path.join(self.work_dir, 'critic1_{}_{}.pth'.format(str(self.cfg.agent.name),self._global_step))
                    torch.save(self.agent.critic, path)
                path = os.path.join(self.work_dir, 'actor1_{}_{}.pth'.format(str(self.cfg.agent.name),self._global_step))
                torch.save(self.agent.actor, path)
            


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
    from pretrain_pixel_gc_only import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()

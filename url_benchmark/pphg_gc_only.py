import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
import itertools
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['HYDRA_FULL_ERROR']='1'

import seaborn as sns; sns.set_theme()
from pathlib import Path
import torch.nn.functional as F
import hydra
import numpy as np
import torch
import wandb
from dm_env import specs
import pandas as pd
import dmc
import utils
from torch import distributions as pyd
from scipy.spatial.distance import cdist
from logger import Logger, save
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_replay_buffer, ndim_grid, make_replay_offline
import matplotlib.pyplot as plt
from video import TrainVideoRecorder, VideoRecorder
from dmc_benchmark import PRIMAL_TASKS
import glob
import natsort
import re

torch.backends.cudnn.benchmark = True

def make_agent(obs_type, obs_spec, action_spec, goal_shape, num_expl_steps, cfg, lr=.0001, hidden_dim=1024, num_protos=512, 
        update_gc=2, gc_only=False, offline=False, tau=.1, num_iterations=3, feature_dim=50, pred_dim=128, proj_dim=512, 
        batch_size=1024, update_proto_every=10, lagr=.2, margin=.5, lagr1=.2, lagr2=.2, lagr3=.3, stddev_schedule=.2, 
        stddev_clip=.3, update_proto=2, stddev_schedule2=.2, stddev_clip2=.3, update_enc_proto=False, update_enc_gc=False, update_proto_opt=True,
        normalize=False, normalize2=False):


    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    cfg.goal_shape = goal_shape
    cfg.lr = lr
    cfg.hidden_dim = hidden_dim
    cfg.num_protos=num_protos
    cfg.tau = tau

    if cfg.name.startswith('proto'):
        cfg.update_gc=update_gc
    cfg.offline=offline
    cfg.gc_only=gc_only
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
    if cfg.name=='protox':
        cfg.lagr1 = lagr1
        cfg.lagr2 = lagr2
        cfg.lagr3 = lagr3
        
    cfg.update_proto_every=update_proto_every
    cfg.stddev_schedule2 = stddev_schedule2
    cfg.stddev_clip2 = stddev_clip2
    cfg.update_enc_proto = update_enc_proto
    cfg.update_enc_gc = update_enc_gc
    cfg.update_proto_opt = update_proto_opt
    cfg.normalize = normalize
    cfg.normalize2 = normalize2
    print('shape', obs_spec.shape)
    return hydra.utils.instantiate(cfg)

def get_state_embeddings(agent, states):
    with torch.no_grad():
        s = agent.encoder(states)
        s = agent.predictor(s)
        s = agent.projector(s)
        s = F.normalize(s, dim=1, p=2)
    return s


def heatmaps(self, env, model_step, replay_dir2, goal,model_step_lb=False,gc=False,proto=False):
    #this only works for 2D mazes
    
    if gc:

        heatmap = self.replay_storage1.state_visitation_gc

        plt.clf()
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        ax.set_title(model_step)

        plt.savefig(f"./{model_step}_gc_heatmap.png")
        wandb.save(f"./{model_step}_gc_heatmap.png")


        #heatmap_pct = self.replay_storage1.state_visitation_gc_pct
        #
        #plt.clf()
        #fig, ax = plt.subplots(figsize=(10,10))
        #labels = np.round(heatmap_pct.T/heatmap_pct.sum()*100, 1)
        #sns.heatmap(np.log(1 + heatmap_pct.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        #ax.set_title(model_step)

        #import IPython as ipy; ipy.embed(colors='neutral')
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
        #labels = np.round(goal_matrix.T/goal_matrix.sum()*100, 1)
        sns.heatmap(np.log(1 + goal_matrix.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        ax.set_title(model_step)

        plt.savefig(f"./{model_step}_goal_state_heatmap.png")
        wandb.save(f"./{model_step}_goal_state_heatmap.png")
        
    if proto:

        heatmap = self.replay_storage.state_visitation_proto

        plt.clf()
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        ax.set_title(model_step)

        plt.savefig(f"./{model_step}_proto_heatmap.png")
        wandb.save(f"./{model_step}_proto_heatmap.png")
        
        heatmap = self.proto_goals_matrix
        
        plt.clf()
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        ax.set_title(model_step)

        plt.savefig(f"./{model_step}_proto_goal_heatmap.png")
        wandb.save(f"./{model_step}_proto_goal_heatmap.png")

        ########################################################
        #exploration moving avg
        indices=[5,10,20,50]
        sets = [self.mov_avg_5, self.mov_avg_10, self.mov_avg_20,
                self.mov_avg_50]
        
        if self.global_step%100000==0:
                    
            plt.clf()
            fig, ax = plt.subplots(figsize=(15,5))
            labels = ['mov_avg_5', 'mov_avg_10', 'mov_avg_20', 'mov_avg_50']

            for ix,x in enumerate(indices):
                ax.plot(np.arange(0,sets[ix].shape[0]), sets[ix], label=labels[ix])
            ax.legend()

            plt.savefig(f"proto_moving_avg_{model_step}.png")
            wandb.save(f"proto_moving_avg_{model_step}.png")
       
        ##########################################################
        #reward moving avg
        sets = [self.r_mov_avg_5, self.r_mov_avg_10, self.r_mov_avg_20,
                self.r_mov_avg_50]
        
        if self.global_step%100000==0:
                    
            plt.clf()
            fig, ax = plt.subplots(figsize=(15,5))
            labels = ['mov_avg_5', 'mov_avg_10', 'mov_avg_20', 'mov_avg_50']

            for ix,x in enumerate(indices):
                ax.plot(np.arange(0,sets[ix].shape[0]), sets[ix], label=labels[ix])
            ax.legend()

            plt.savefig(f"gc_reward_moving_avg_{model_step}.png")
            wandb.save(f"gc_reward_moving_avg_{model_step}.png")
            
        


        
        
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
                cfg.experiment, cfg.agent.name, cfg.task, cfg.obs_type,
                str(cfg.seed), str(cfg.tmux_session),work_path 
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
        #two different routes for pmm vs. non-pmm envs
        if self.pmm:
            self.no_goal_task = self.cfg.task_no_goal
            idx = np.random.randint(0,400)
            goal_array = ndim_grid(2,20)
            self.first_goal = np.array([goal_array[idx][0], goal_array[idx][1]])
            
            self.train_env1 = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                       cfg.action_repeat, seed=None, goal=self.first_goal)
            print('goal', self.first_goal)
            
            self.train_env_no_goal = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                       cfg.action_repeat, seed=None, goal=None)
            print('no goal task env', self.no_goal_task)

            self.eval_env_no_goal = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                       cfg.action_repeat, seed=None, goal=None)
            
            self.eval_env_goal = dmc.make(self.no_goal_task, 'states', cfg.frame_stack,
                                       1, seed=None, goal=None)
            
            self.eval_env = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                     cfg.action_repeat, seed=None, goal=self.first_goal)
        else:
            self.train_env1 = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                       cfg.action_repeat, seed=None)
            self.train_env = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                                      cfg.action_repeat, seed=None)
            self.eval_env = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                     cfg.action_repeat, seed=None)
        print('irmak', cfg.irmak) 
        if cfg.cassio:
            self.pwd = '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark'
        elif cfg.greene:
            self.pwd = '/vast/nm1874/dm_control_2022/proto_explore/url_benchmark'
        elif cfg.pluto or cfg.irmak:
            self.pwd = '/home/nina/proto_explore/url_benchmark' 
        # create agent

        self.agent = make_agent(cfg.obs_type,
                            self.train_env1.observation_spec(),
                            self.train_env1.action_spec(),
                            (3*self.cfg.frame_stack,84,84),
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

        if self.cfg.agent.name == 'proto_inv':
            self.inv=True
        else:
            self.inv=False
        print('agent', self.cfg.agent.name)    
        # initialize from pretrained
        print('model p', cfg.model_path)
        if cfg.model_path:
            path = self.cfg.model_path.split('/')
            path = Path(self.pwd+'/'.join(path[:-1]))
            print('path', str(self.pwd + cfg.model_path) + str('*.pth'))
            self.agent_path = natsort.natsorted(glob.glob(str(self.pwd + cfg.model_path) + str('*.pth')))
            self.path_lst = [int(re.findall('\d+', x)[-1]) for x in self.agent_path]
            print('path lst', self.path_lst)
            
            assert os.path.isfile(self.agent_path[self.cfg.model_step_index])
            self.pretrained_agent = torch.load(self.agent_path[self.cfg.model_step_index])
            print('agent path', self.agent_path[self.cfg.model_step_index])
            print('pretrained_agent', vars(self.pretrained_agent))
            self.agent.init_encoder_from(self.pretrained_agent.encoder)
            
        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env1.observation_spec(),
                      self.train_env1.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        if self.cfg.offline_gc==False:
            # create data storage
            self.replay_storage1 = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer1')

 

        # create replay buffer
 
            
        if self.cfg.offline_gc:
            print('offline buffer')
            print('inv', self.inv)
            self.replay_loader1 = make_replay_offline(self.eval_env,
                                                    path / 'buffer2' / 'buffer_copy',
                                                    cfg.replay_buffer_gc,
                                                    cfg.batch_size_gc,
                                                    cfg.replay_buffer_num_workers,
                                                    cfg.discount,
                                                    offset=100,
                                                    goal=True,
                                                    relabel=False,
                                                    replay_dir2=False,
                                                    obs_type='state',
                                                    offline=False,
                                                    nstep=self.cfg.nstep,
                                                    eval=False,
                                                    inv=self.inv,
                                                    goal_offset=self.cfg.goal_offset,
                                                    pmm=self.pmm,
                                                    model_step=10000)

        
        elif self.cfg.combine_storage_gc:
            print('combine')
            self.replay_loader1 = make_replay_loader(self.replay_storage1,
                                                    True,
                                                    cfg.replay_buffer_gc,
                                                    cfg.batch_size_gc,
                                                    cfg.replay_buffer_num_workers,
                                                    False, cfg.nstep, cfg.discount,
                                                    True, cfg.hybrid_gc,cfg.obs_type,
                                                    cfg.hybrid_pct, sl=cfg.sl,
                                                    replay_dir2=path / 'buffer2' / 'buffer_copy',
                                                    loss=cfg.loss_gc, test=cfg.test,
                                                    tile=cfg.frame_stack,
                                                    pmm=self.pmm,
                                                    obs_shape=self.train_env1.physics.state().shape[0],
                                                    general=True,
                                                    inv=self.inv,
                                                    goal_offset=self.cfg.goal_offset)
 

        self._replay_iter1 = None

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
        self.distance_goal_init = {}
        
        if self.cfg.proto_goal_intr:
            self.dim=10
        else:
            self.dim=self.agent.protos.weight.data.shape[0]

        self.proto_goals = np.zeros((self.dim, 3*self.cfg.frame_stack, 84, 84))
        self.proto_goals_state = np.zeros((self.dim, self.train_env1.physics.get_state().shape[0]))
        self.proto_goals_dist = np.zeros((self.dim, 1))
        self.proto_goals_matrix = np.zeros((60,60))
        self.proto_goals_id = np.zeros((self.dim, 2))
        self.actor1=True
        self.final_df = pd.DataFrame(columns=['avg', 'med', 'max', 'q7', 'q8', 'q9'])
        self.reached_goals=np.zeros((2,60,60))
        self.origin = None
        self.proto_explore=False
        self.proto_explore_count=0
        self.gc_explore=False
        self.gc_explore_count=0
        self.previous_matrix = None
        self.current_matrix = None
        self.v_queue_ptr = 0 
        self.v_queue = np.zeros((2000,))
        self.r_queue_ptr = 0 
        self.r_queue = np.zeros((2000,))
        self.count=0
        self.mov_avg_5 = np.zeros((2000,))
        self.mov_avg_10 = np.zeros((2000,))
        self.mov_avg_20 = np.zeros((2000,))
        self.mov_avg_50 = np.zeros((2000,))
        self.mov_avg_100 = np.zeros((2000,))
        self.mov_avg_200 = np.zeros((2000,))
        self.mov_avg_500 = np.zeros((2000,))
        self.r_mov_avg_5 = np.zeros((2000,))
        self.r_mov_avg_10 = np.zeros((2000,))
        self.r_mov_avg_20 = np.zeros((2000,))
        self.r_mov_avg_50 = np.zeros((2000,))
        self.r_mov_avg_100 = np.zeros((2000,))
        self.r_mov_avg_200 = np.zeros((2000,))
        self.r_mov_avg_500 = np.zeros((2000,))
        self.unreached_goals = np.empty((0,self.train_env1.physics.get_state().shape[0]))
        self.proto_last_explore=0
        self.current_init = np.empty((0,self.train_env1.physics.get_state().shape[0]))
        self.unreached = False
        self.gc_init = False
        goal_array = ndim_grid(2,7)
        self.eval_reached = np.array([goal_array[6]])
    
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

    
    def eval_proto(self, evaluate=False):
        
        if evaluate:
            
            path = self.cfg.model_path.split('/')
            path = Path(self.pwd+'/'.join(path[:-1]))
            replay_buffer = make_replay_offline(self.eval_env,
                                                    path / 'buffer1' / 'buffer_copy',
                                                    500000,
                                                    0,
                                                    0,
                                                    .99,
                                                    goal=False,
                                                    relabel=False,
                                                    replay_dir2=False,
                                                    obs_type = 'pixels',
                                                   
                                                
                                                    )

            
            state, actions, rewards, goal_states, eps, index = replay_buffer.parse_dataset(goal_state=True) 
            
            df = pd.DataFrame({'s0':state[:,0], 's1':state[:,1],'r':rewards[:,0], 'g0':goal_states[:,0], 'g1':goal_states[:,1], 'e':eps})
            df['eps'] = [x.split('/')[-1] for x in df['e']]
            df1 = pd.DataFrame()
            df1['g0'] = df.groupby('eps')['g0'].first()
            df1['g1'] = df.groupby('eps')['g1'].first()
            df1['r'] = df.groupby('eps')['r'].sum()
            df1=df1[df1['r']>100]
            df1=df1.reset_index(drop=True)
            self.current_init=df1[['g0', 'g1']].to_numpy()
            self.current_init=self.current_init[sorted(np.unique(self.current_init, return_index=True, axis=0)[1])]
            self.current_init=np.concatenate((self.current_init, np.zeros((self.current_init.shape[0], 2))), axis=1)


        else:
            
          
            if self.global_step in self.path_lst:
                agent_idx = self.path_lst.index(self.global_step)
                assert os.path.isfile(self.agent_path[agent_idx])
                self.pretrained_agent = torch.load(self.agent_path[agent_idx])
           
            
            
            if self.global_step%100000==0 and self.pmm and self.cfg.offline_gc==False:
                heatmaps(self, self.eval_env, self.global_step, False, True, model_step_lb=False,gc=True,proto=True)

            ########################################################################
            #how should we measure exploration in non-pmm w/o heatmaps
            
            ############################################################
            #offline loader is not adding in the replay_dir2 right now, change this !!!!!!!!!!!!! 
            path = self.cfg.model_path.split('/')
            path = Path(self.pwd+'/'.join(path[:-1])) 
            replay_buffer = make_replay_offline(self.eval_env,
                                                    path / 'buffer2' / 'buffer_copy',
                                                    500000,
                                                    0,
                                                    0,
                                                    .99,
                                                    goal=False,
                                                    relabel=False,
                                                    replay_dir2=False,
                                                    obs_type = 'pixels',
                                                    model_step = self.global_step
                                                    )


            state, actions, rewards, eps, index = replay_buffer.parse_dataset()
            state = state.reshape((state.shape[0], self.train_env1.physics.get_state().shape[0]))
            

        while self.proto_goals.shape[0] < self.dim:
            self.proto_goals = np.append(self.proto_goals, np.zeros((1,3*self.cfg.frame_stack,84,84)), axis=0)
            self.proto_goals_state = np.append(self.proto_goals_state, np.array([[0., 0., 0., 0.]]), axis=0)
            self.proto_goals_dist = np.append(self.proto_goals_dist, np.array([[0.]]), axis=0)
        protos = self.pretrained_agent.protos.weight.data.detach().clone()

        num_sample=600 
        idx = np.random.randint(0, state.shape[0], size=num_sample)
        state=state[idx]
        state=state.reshape(num_sample,self.train_env1.physics.get_state().shape[0])
        a = state

        encoded = []
        proto = []
        actual_proto = []
        lst_proto = []

        for x in idx:
            fn = eps[x]
            idx_ = index[x]
            ep = np.load(fn)
            #pixels.append(ep['observation'][idx_])

            with torch.no_grad():
                obs = ep['observation'][idx_]
                #import IPython as ipy; ipy.embed(colors='neutral')
                ##################
                #why is there an extra dim here
                if obs.shape[0]!=1:
                    obs = torch.as_tensor(obs.copy(), device=self.device).unsqueeze(0)
                else:
                    obs = torch.as_tensor(obs.copy(), device=self.device)
                if self.cfg.sl==False:
                    z = self.pretrained_agent.encoder(obs)
                else:
                    
                    z = self.pretrained_agent.encoder2(obs)
                encoded.append(z)
                z = self.pretrained_agent.predictor(z)
                z = self.pretrained_agent.projector(z)
                if self.cfg.normalize:
                    z = F.normalize(z, dim=1, p=2)
                proto.append(z)
                sim = self.pretrained_agent.protos(z)
                idx_ = sim.argmax()
                actual_proto.append(protos[idx_][None,:])

        encoded = torch.cat(encoded,axis=0)
        proto = torch.cat(proto,axis=0)
        actual_proto = torch.cat(actual_proto,axis=0)

        sample_dist = torch.norm(proto[:,None,:] - proto[None,:, :], dim=2, p=2)

#         if self.pmm:

#             df = pd.DataFrame()
#             df['x'] = a[:,0].round(2)
#             df['y'] = a[:,1].round(2)
#             df['r'] = sample_dist[0].clone().detach().cpu().numpy()
#             result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['r']
#             result.fillna(0, inplace=True)
#             sns.heatmap(result, cmap="Blues_r",fmt='.2f', ax=ax).invert_yaxis()
#             ax.set_xticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_xticklabels()])
#             ax.set_yticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_yticklabels()])
#             ax.set_title('{}, {}'.format(self.global_step, a[0,:2]))  

#             plt.savefig(f"./{self.global_step}_dist_heatmap.png")
#             wandb.save(f"./{self.global_step}_dist_heatmap.png")

        proto_dist = torch.norm(protos[:,None,:] - proto[None,:, :], dim=2, p=2)

        all_dists_proto, _proto = torch.topk(proto_dist, 10, dim=1, largest=False)

        p = _proto.clone().detach().cpu().numpy()

        if self.cfg.proto_goal_intr:

            goal_dist, goal_indices = self.eval_intrinsic(encoded, a)
            dist_arg = self.proto_goals_dist.argsort(axis=0)

            for ix,x in enumerate(goal_dist.clone().detach().cpu().numpy()):

                if x > self.proto_goals_dist[dist_arg[ix]]:

                    self.proto_goals_dist[dist_arg[ix]] = x
                    closest_sample = goal_indices[ix].clone().detach().cpu().numpy()

                    ##################################
                    #may need to debug this 
#                     fn = eps[closest_sample]
#                     idx_ = index[closest_sample]
#                     ep = np.load(fn)

#                     with torch.no_grad():
#                         obs = ep['observation'][idx_]

#                     self.proto_goals[dist_arg[ix]] = obs

                    self.proto_goals_state[dist_arg[ix]] = a[closest_sample]

                    with self.eval_env_no_goal.physics.reset_context():

                        self.eval_env_no_goal.physics.set_state(a[closest_sample][0])

                    img = self.eval_env_no_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get(self.cfg.domain, 0))

                    img = np.transpose(img, (2,0,1))
                    img = np.tile(img, (self.cfg.frame_stack,1,1))
                    self.proto_goals[dist_arg[ix]]=img

            print('proto goals', self.proto_goals_state)
            print('proto dist', self.proto_goals_dist)

        elif self.cfg.proto_goal_random:

            closest_sample = _proto[:, 0].detach().clone().cpu().numpy()

        ################################################
#             for ix, x in enumerate(closest_sample):

#                 fn = eps[x]
#                 idx_ = index[x]
#                 ep = np.load(fn)
#                 #pixels.append(ep['observation'][idx_])

#                 with torch.no_grad():
#                     obs = ep['observation'][idx_]

#                 self.proto_goals[ix] = obs
            self.proto_goals_state = a[closest_sample]

            for ix in range(self.proto_goals_state.shape[0]):           
                with self.eval_env_no_goal.physics.reset_context():                 
                    self.eval_env_no_goal.physics.set_state(self.proto_goals_state[ix])

                img = self.eval_env_no_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get(self.cfg.domain, 0))
                img = np.transpose(img, (2,0,1))
                img = np.tile(img, (self.cfg.frame_stack,1,1))
                self.proto_goals[ix] = img

        if self.pmm:

            filenames=[]
            plt.clf()
            fig, ax = plt.subplots()
            dist_np = np.empty((protos.shape[1], _proto.shape[1], 2))
            for ix in range(protos.shape[0]):
                txt=''
                df = pd.DataFrame()
                count=0
                for i in range(a.shape[0]+1):
                    if i!=a.shape[0]:
                        df.loc[i,'x'] = a[i,0]
                        df.loc[i,'y'] = a[i,1]
                        if i in _proto[ix,:]:
                            df.loc[i, 'c'] = str(ix+1)
                            dist_np[ix,count,0] = a[i,0]
                            dist_np[ix,count,1] = a[i,1]
                            count+=1

                        elif ix==0 and (i not in _proto[ix,:]):
                            #color all samples blue
                            df.loc[i,'c'] = str(0)

                palette = {
                                '0': 'tab:blue',
                                    '1': 'tab:orange',
                                    '2': 'black',
                                    '3':'silver',
                                    '4':'green',
                                    '5':'red',
                                    '6':'purple',
                                    '7':'brown',
                                    '8':'pink',
                                    '9':'gray',
                                    '10':'olive',
                                    '11':'cyan',
                                    '12':'yellow',
                                    '13':'skyblue',
                                    '14':'magenta',
                                    '15':'lightgreen',
                                    '16':'blue',
                                    '17':'lightcoral',
                                    '18':'maroon',
                                    '19':'saddlebrown',
                                    '20':'peru',
                                    '21':'tan',
                                    '22':'darkkhaki',
                                    '23':'darkolivegreen',
                                    '24':'mediumaquamarine',
                                    '25':'lightseagreen',
                                    '26':'paleturquoise',
                                    '27':'cadetblue',
                                    '28':'steelblue',
                                    '29':'thistle',
                                    '30':'slateblue',
                                    '31':'hotpink',
                                    '32':'papayawhip'
                        }
                ax=sns.scatterplot(x="x", y="y",
                          hue="c",palette=palette,
                          data=df,legend=True)
                sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
                #ax.set_title("\n".join(wrap(txt,75)))

                file1= self.work_dir / f"10nn_actual_prototypes_{self.global_step}.png"
                plt.savefig(file1)
                wandb.save(f"10nn_actual_prototypes_{self.global_step}.png")

        ########################################################################
        #implement tsne for non-pmm prototype eval?
        if self.global_step%100000==0 and self.pmm==False: 
            for ix in range(self.proto_goals_state.shape[0]):

                with self.eval_env.physics.reset_context():
                    self.eval_env.physics.set_state(self.proto_goals_state[ix])

                plt.clf()
                img = self.eval_env._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get(self.cfg.domain, 0))
                plt.imsave(f"goals_{ix}_{self.global_step}.png", img)
                wandb.save(f"goals_{ix}_{self.global_step}.png")
                
        #delete goals that have been reached
        if self.current_init.shape[0]>0:
            index=np.where(((np.linalg.norm(self.proto_goals_state[:,None,:] - self.current_init[None,:,:],axis=-1, ord=2))<.05))
            index = np.unique(index[0])
            print('delete goals', self.proto_goals_state[index])
            self.proto_goals = np.delete(self.proto_goals, index,axis=0)
            self.proto_goals_state = np.delete(self.proto_goals_state, index,axis=0)
            self.proto_goals_dist = np.delete(self.proto_goals_dist, index,axis=0)
            index=np.where((self.proto_goals_state==0.).all(axis=1))[0]
            self.proto_goals = np.delete(self.proto_goals, index,axis=0)
            self.proto_goals_state = np.delete(self.proto_goals_state, index,axis=0)
            self.proto_goals_dist = np.delete(self.proto_goals_dist, index,axis=0)
            print('current goals', self.proto_goals_state) 
            assert self.proto_goals_state.shape[0] == self.proto_goals.shape[0] == self.proto_goals_dist.shape[0]
        else:
            print('current goals', self.proto_goals_state) 



    def eval_pmm(self):
        df = pd.DataFrame(columns=['x','y','r'], dtype=np.float64)
        print('eval reached', self.eval_reached)
        for idx in range(self.eval_reached.shape[0]):
            goal_array = ndim_grid(2,8)
            init = self.eval_reached[idx]
            goal_index = np.where(np.all(goal_array==init,axis=1))[0]
            goal_array = torch.tensor(goal_array)
            g_to_g = torch.norm(goal_array[:, None, :] - goal_array[None, :, :], dim=2, p=2)
            all_dists, _ = torch.topk(g_to_g , 8, dim=1, largest=False)
            goal_array = goal_array[_[goal_index]].cpu().numpy()[0]
            print('g', goal_array)
            for ix, x in enumerate(goal_array):

                step, episode, total_reward = 0, 0, 0
                #goal_pix, goal_state = self.sample_goal_uniform(eval=True)
                goal_state = x
                print('goal_state', x)
                print('init', init)
                self.eval_env = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                        self.cfg.action_repeat, seed=None, goal=goal_state, init_state=init)
                self.eval_env_goal = dmc.make(self.no_goal_task, 'states', self.cfg.frame_stack,
                        self.cfg.action_repeat, seed=None, goal=None)
                eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
                meta = self.agent.init_meta()

                while eval_until_episode(episode):
                    time_step = self.eval_env.reset()
                    self.eval_env_no_goal = dmc.make(self.no_goal_task, self.cfg.obs_type, self.cfg.frame_stack,
                        self.cfg.action_repeat, seed=None, goal=None, init_state=time_step.observation['observations'][:2]) 
                    time_step_no_goal = self.eval_env_no_goal.reset()

                    with self.eval_env_goal.physics.reset_context():
                        time_step_goal = self.eval_env_goal.physics.set_state(np.array([goal_state[0], goal_state[1], 0, 0]))
                    time_step_goal = self.eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
                    time_step_goal = np.transpose(time_step_goal, (2,0,1))
                    self.video_recorder.init(self.eval_env, enabled=(episode == 0))

                    while step!=self.cfg.episode_length:
                        with torch.no_grad(), utils.eval_mode(self.agent):
                            if self.cfg.goal:
                                action = self.agent.act(time_step_no_goal.observation['pixels'],
                                                    time_step_goal,
                                                    meta,
                                                    self._global_step,
                                                    eval_mode=True,
                                                    tile=1,
                                                    general=True)
                            else:
                                action = self.agent.act(time_step.observation,
                                                    meta,
                                                    self._global_step,
                                                    eval_mode=True,
                                                    tile=self.cfg.frame_stack)
                        time_step = self.eval_env.step(action)
                        time_step_no_goal = self.eval_env_no_goal.step(action)
                        #time_step_goal = self.eval_env_goal.step(action)
                        self.video_recorder.record(self.eval_env)
                        total_reward += time_step.reward
                        step += 1

                    episode += 1

                    
                    #if ix%10==0:
                    self.video_recorder.save(f'{self.global_frame}_{ix}.mp4')

                    if self.cfg.eval:
                        save(str(self.work_dir)+'/eval_{}.csv'.format(model.split('.')[-2].split('/')[-1]), [[x.cpu().detach().numpy(), total_reward, time_step.observation[:2], step]])

                    else:
                        save(str(self.work_dir)+'/eval_{}.csv'.format(self._global_step), [[goal_state, total_reward, time_step.observation['observations'], step]])

                    if total_reward > 10*self.cfg.num_eval_episodes:
                        self.eval_reached = np.append(self.eval_reached, x[None,:], axis=0)
                        self.eval_reached = np.unique(self.eval_reached, axis=0)

                df.loc[ix, 'x'] = x[0].round(2)
                df.loc[ix, 'y'] = x[1].round(2)
                df.loc[ix, 'r'] = total_reward

                print('r', total_reward)

        result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['r']/2
        result.fillna(0, inplace=True)
        print('result', result)
        plt.clf()
        fig, ax = plt.subplots()
        sns.heatmap(result, cmap="Blues_r").invert_yaxis()
        ax.set_xticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_xticklabels()])
        ax.set_yticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_yticklabels()])
        plt.savefig(f"./{self.global_step}_heatmap_goal.png")
        wandb.save(f"./{self.global_step}_heatmap_goal.png")


    def eval_intrinsic(self, encoded, states):

        with torch.no_grad():
            reward = self.agent.compute_intr_reward(encoded, None, self._global_step, eval=True)

        if self.cfg.proto_goal_intr:
            #import IPython as ipy; ipy.embed(colors='neutral') 
            r, _ = torch.topk(reward,5,largest=True, dim=0)

        df = pd.DataFrame()
        df['x'] = states[:,0].round(2)
        df['y'] = states[:,1].round(2)
        df['r'] = reward.detach().clone().cpu().numpy()
        result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['r'].round(2)
        #import IPython as ipy; ipy.embed(colors='neutral')
        result.fillna(0, inplace=True)
        plt.clf()
        fig, ax = plt.subplots(figsize=(10,6))

        sns.heatmap(result, cmap="Blues_r",fmt='.2f', ax=ax).invert_yaxis()
        ax.set_xticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_xticklabels()])
        ax.set_yticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_yticklabels()])
        ax.set_title(self.global_step)
        plt.savefig(f"./{self.global_step}_intr_reward.png")
        wandb.save(f"./{self.global_step}_intr_reward.png")

        if self.cfg.proto_goal_intr:
            return r, _

    
    
    def evaluate(self):
        self.logger.log('eval_total_time', self.timer.total_time(),
                        self.global_frame)
        if self.cfg.debug:
            self.eval()
        elif self.pmm:
            self.eval_proto()
            #self.eval_pmm()
        else:
            self.eval_proto()
            #if self.global_step > 20000 and self.global_step%10000==0:
            #    print('2')
            #    self.eval()
           
            
            
    def make_env(self, actor1, init_idx, goal_state, pmm):
        

        if pmm:
            if goal_state is not None:
                goal_state = goal_state[:2]
            if init_idx is None:
                
                init_state = np.random.uniform(.25,.29,size=(2,))
                init_state[0] = init_state[0]*(-1)
            
            else: 
                
                init_state = self.current_init[init_idx,:2]
            

            self.train_env1 = dmc.make(self.cfg.task, self.cfg.obs_type,
                                           self.cfg.frame_stack,self.cfg.action_repeat,
                                           seed=None, goal=goal_state, init_state = init_state) 

            time_step1 = self.train_env1.reset()


            self.train_env_no_goal = dmc.make(self.no_goal_task, self.cfg.obs_type, self.cfg.frame_stack,
                                            self.cfg.action_repeat, seed=None, goal=None, 
                                            init_state=time_step1.observation['observations'][:2])

            time_step_no_goal = self.train_env_no_goal.reset()
 
            self.origin = init_state
        else:
            
            if init_idx is None:
                
                time_step = self.train_env.reset()
                self.origin = self.train_env.physics.get_state()
            
            else: 
                
                time_step = self.train_env.reset()
                ##can't reset now
                with self.train_env.physics.reset_context():
                    self.train_env.physics.set_state(self.current_init[init_idx])
                    
                act_ = np.zeros(self.train_env.action_spec().shape, self.train_env.action_spec().dtype)
                time_step = self.train_env.step(act_)

                self.origin = self.current_init[init_idx]
        
        return time_step1, time_step_no_goal

    
    
    def save_stats(self):
        
        #record changes in proto heatmap
        if self.global_step%1000==0 and self.global_step>5000:
            
            if self.pmm:

                total_v = np.count_nonzero(self.replay_storage.state_visitation_proto)
                print('total visitation', total_v)
                v_ptr = self.v_queue_ptr
                self.v_queue[v_ptr] = total_v
                self.v_queue_ptr = (v_ptr+1) % self.v_queue.shape[0]

                indices=[5,10,20,50]
                sets = [self.mov_avg_5, self.mov_avg_10, self.mov_avg_20,
                        self.mov_avg_50]

                for ix,x in enumerate(indices):
                    if self.v_queue_ptr-x<0:
                        lst = np.concatenate([self.v_queue[:self.v_queue_ptr], self.v_queue[self.v_queue_ptr-x:]], axis=0)
                        sets[ix][self.count]=lst.mean()
                    else:
                        sets[ix][self.count]=self.v_queue[self.v_queue_ptr-x:self.v_queue_ptr].mean()
                
                total_r = np.count_nonzero(self.replay_storage1.reward_matrix)
                print('total reward', total_r)
                r_ptr = self.r_queue_ptr
                self.r_queue[r_ptr] = total_r
                self.r_queue_ptr = (r_ptr+1) % self.r_queue.shape[0]
                
                sets = [self.r_mov_avg_5, self.r_mov_avg_10, self.r_mov_avg_20,
                        self.r_mov_avg_50]

                for ix,x in enumerate(indices):
                    if self.r_queue_ptr-x<0:
                        lst = np.concatenate([self.r_queue[:self.r_queue_ptr], self.r_queue[self.r_queue_ptr-x:]], axis=0)
                        sets[ix][self.count]=lst.mean()
                    else:
                        sets[ix][self.count]=self.r_queue[self.r_queue_ptr-x:self.r_queue_ptr].mean()
                        

                self.count+=1

        #save stats
        #change to 100k when not testing
        if self.global_step%100000==0:
            df = pd.DataFrame()
            if self.pmm:
                df['mov_avg_5'] = self.mov_avg_5
                df['mov_avg_10'] = self.mov_avg_10
                df['mov_avg_20'] = self.mov_avg_20
                df['mov_avg_50'] = self.mov_avg_50
            df['r_mov_avg_5'] = self.r_mov_avg_5
            df['r_mov_avg_10'] = self.r_mov_avg_10
            df['r_mov_avg_20'] = self.r_mov_avg_20
            df['r_mov_avg_50'] = self.r_mov_avg_50
            path = os.path.join(self.work_dir, 'exploration_{}_{}.csv'.format(str(self.cfg.agent.name),self._global_step))
            df.to_csv(path, index=False)
            

    
    def sample_goal(self):
        
        s = self.proto_goals.shape[0]
        num = s+self.unreached_goals.shape[0]
        idx = np.random.randint(num)

        if idx >= s:

            goal_idx = idx-s
            goal_state = self.unreached_goals[goal_idx]
            with self.eval_env.physics.reset_context():
                self.eval_env.physics.set_state(goal_state)
            goal_pix = self.eval_env._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
            goal_pix = np.transpose(goal_pix, (2,0,1))
            goal_pix = np.tile(goal_pix, (self.cfg.frame_stack,1,1))
            self.unreached=True


        else:
            goal_idx = idx
            goal_state = self.proto_goals_state[goal_idx]
            goal_pix = self.proto_goals[goal_idx]
            
        return goal_idx, goal_state, goal_pix
            
        
    def calc_reward(self, pix, goal_pix):
        
        if self.cfg.ot_reward:
                        
            reward = self.agent.ot_rewarder(pix, goal_pix, self.global_step)

#                     elif self.cfg.dac_reward:

#                         reward = self.agent.dac_rewarder(time_step1.observation['pixels'], action1)

        elif self.cfg.neg_euclid:

            with torch.no_grad():
                obs = pix
                obs = torch.as_tensor(obs.copy(), device=self.device).unsqueeze(0)
                z1 = self.agent.encoder(obs)
                z1 = self.agent.predictor(z1)
                z1 = self.agent.projector(z1)
                z1 = F.normalize(z1, dim=1, p=2)

                goal = torch.as_tensor(goal_pix, device=self.device).unsqueeze(0).int()

                z2 = self.agent.encoder(goal)
                z2 = self.agent.predictor(z2)
                z2 = self.agent.projector(z2)
                z2 = F.normalize(z2, dim=1, p=2)
                
            reward = -torch.norm(z1-z2, dim=-1, p=2).item()
            
        elif self.cfg.neg_euclid_state:

            reward = -np.linalg.norm(self.train_env1.physics.get_state() - goal_state, axis=-1, ord=2)
            
        elif self.cfg.actionable:

            print('not implemented yet: actionable_reward')
    
    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        log_every_step = utils.Every(self.cfg.log_every_steps)
        episode_step, episode_reward = 0, 0
        
        meta = self.agent.init_meta() 
         
        metrics = None
        
        goal_idx = 0 
        
        if self.pmm==False:
            time_step_no_goal = None
            
        if self.cfg.model_path and self.cfg.offline_gc==False:
            self.eval_proto()
        
        while train_until_step(self.global_step):
            if self.cfg.offline_gc:
                # try to evaluate
                if eval_every_step(self.global_step+1):
                    self.logger.log("eval_total_time", self.timer.total_time(), self.global_step)
                    print('gs', self.global_step)
                    if (self.global_step+1)%100000==0:
                        self.eval_pmm()

                metrics = self.agent.update(self.replay_iter1, self.global_step, actor1=True)
                self.logger.log_metrics(metrics, self.global_step, ty="train")
                if log_every_step(self.global_step):
                    elapsed_time, total_time = self.timer.reset()
                    
                    with self.logger.log_and_dump_ctx(self.global_step, ty="train") as log:
                        log("fps", self.cfg.log_every_steps / elapsed_time)
                        log("total_time", total_time)
                        log("step", self.global_step)

#                 if global_step%100000==0:
#                     path = os.path.join(work_dir, 'optimizer_goal_{}_{}.pth'.format(global_index,global_step))
#                     torch.save(agent,path)
                self._global_step += 1
            
            else:


                #reload data every 100k steps
                if self.global_step%100000==0 or self.global_step==self.cfg.num_seed_frames or (self.global_step%10000==0 and self.global_step<100000):
                    self.replay_loader1 = make_replay_offline(self.eval_env,
                                                        path / 'buffer2' / 'buffer_copy',
                                                        cfg.replay_buffer_gc,
                                                        cfg.batch_size_gc,
                                                        cfg.replay_buffer_num_workers,
                                                        cfg.discount,
                                                        offset=100,
                                                        goal=True,
                                                        relabel=False,
                                                        replay_dir2=False,
                                                        obs_type='state',
                                                        offline=False,
                                                        nstep=self.cfg.nstep,
                                                        eval=False,
                                                        load_every=0,
                                                        inv=self.inv,
                                                        goal_offset=self.cfg.goal_offset,
                                                        pmm=self.pmm,
                                                        model_step=self.global_step)


                    episode_step=0
                    episode_reward=0

                    self.actor1=True

                    #render first goal

                    goal_pix = self.proto_goals[0]
                    goal_state = self.proto_goals_state[0]


                    time_step1, time_step_no_goal = self.make_env(actor1=self.actor1, init_idx=None , goal_state=goal_state, pmm=self.pmm)  
                    #non-pmm we just use current state

                    meta = self.agent.update_meta(meta, self._global_step, time_step1)

                    self.replay_storage1.add_goal_general(time_step1, self.train_env1.physics.get_state(), meta, 
                                                              goal_pix, goal_state, 
                                                              time_step_no_goal, True)

                    if self.cfg.model_path==False:
                        self.eval_proto()
                self.save_stats()



                if (time_step1.last() and self.actor1):
                    print('last')
                    self._global_episode += 1
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

                    if self.cfg.obs_type =='pixels' and self.actor1:
                        self.replay_storage1.add_goal_general(time_step1, self.train_env1.physics.get_state(), meta, 
                                                              goal_pix, goal_state, 
                                                              time_step_no_goal, True, last=True)

                    self.unreached=False

                    # try to save snapshot
                    ##################################
                    #check why this isn't working
                    if self.global_frame in self.cfg.snapshots:
                        self.save_snapshot()

                    episode_step = 0
                    episode_reward = 0

                # try to evaluate
                if eval_every_step(self.global_step) and self.global_step!=0:
                    self.evaluate()

                if episode_step== 0 and self.global_step!=0:

                    print('gc policy')
                    assert self.actor1==True
                    assert self.proto_goals_state.shape[0] == self.proto_goals.shape[0] == self.proto_goals_dist.shape[0]

                    goal_idx, goal_state, goal_pix = self.sample_goal()

                    if len(self.current_init) != 0:

                        if self.current_init.shape[0] > 2:

                            chance = np.random.uniform()
                            if chance < .8:
                                init_idx = -np.random.randint(1,4)
                            else:
                                init_idx = np.random.randint(self.current_init.shape[0])

                        elif len(self.current_init) > 0:
                            init_idx = -1                    

                        time_step1, time_step_no_goal = self.make_env(actor1=self.actor1, init_idx=init_idx, goal_state=goal_state, pmm=self.pmm)

                    else:

                        time_step1, time_step_no_goal = self.make_env(actor1=self.actor1, init_idx=None, goal_state=goal_state, pmm=self.pmm)

                    meta = self.agent.update_meta(meta, self._global_step, time_step1) 
                    print('time step', time_step1.observation['observations'])
                    print('sampled goal', goal_state)

                   #unreached_goals : array of states (not pixel), need to set state & render to get pixel
                    if self.cfg.obs_type == 'pixels' and time_step1.last()==False and episode_step!=self.cfg.episode_length:
                        self.replay_storage1.add_goal_general(time_step1, self.train_env1.physics.get_state(), meta,
                                                              goal_pix, goal_state,
                                                              time_step_no_goal, True)

                # sample action

                with torch.no_grad(), utils.eval_mode(self.agent):
                    if self.cfg.obs_type == 'pixels' and self.pmm:
                        action1 = self.agent.act(time_step_no_goal.observation['pixels'].copy(),
                                                goal_pix,
                                                meta,
                                                self._global_step,
                                                eval_mode=False,
                                                tile=1,
                                                general=True)

                    #non-pmm
                    elif self.cfg.obs_type == 'pixels':
                        action1 = self.agent.act(time_step1.observation['pixels'].copy(),
                                                goal_pix,
                                                meta,
                                                self._global_step,
                                                eval_mode=False,
                                                tile=1,
                                                general=True)


                # take env step
                time_step1 = self.train_env1.step(action1)

                if self.pmm:

                    time_step_no_goal = self.train_env_no_goal.step(action1)

                if self.pmm == False:
                    print('not pmm, calculating reward')
                    #calculate reward
                    reward = self.calc_reward(time_step1.observation['pixels'], goal_pix)
                    time_step1 = time_step1._replace(reward=reward)

                #higher dim envs have -1 as cutoff 
                if time_step1.reward > self.cfg.reward_cutoff:

                    episode_reward += (time_step1.reward + abs(self.cfg.reward_cutoff))

                if self.cfg.obs_type == 'pixels' and time_step1.last()==False and episode_step!=self.cfg.episode_length and ((episode_reward < abs(self.cfg.reward_cutoff*20) and self.pmm==False) or ((episode_reward < self.cfg.pmm_reward_cutoff) and self.pmm)):

                    self.replay_storage1.add_goal_general(time_step1, self.train_env1.physics.get_state(), meta, 
                                                          goal_pix, goal_state, 
                                                          time_step_no_goal, True)




                if time_step1.reward == 0. and self.pmm==False:
                    time_step1 = time_step1._replace(reward=-1)

                if (self.pmm==False and (episode_reward > abs(self.cfg.reward_cutoff*20)) and episode_step>5) or (self.pmm and episode_reward > 100):
                    print('proto_goals', self.proto_goals_state)
                    print('r', episode_reward)
                    idx_x = int(goal_state[0]*100)+29
                    idx_y = int(goal_state[1]*100)+29
                    origin_x = int(self.origin[0]*100)+29
                    origin_y = int(self.origin[1]*100)+29
                    self.reached_goals[0,origin_x, idx_x] = 1
                    self.reached_goals[1,origin_y, idx_y] = 1
                    self.proto_goals_matrix[idx_x,idx_y]+=1 

                    ##############################
                    #add non-pmm later 
                    if self.pmm:
                        self.unreached_goals = np.round(self.unreached_goals,2)
                        print('u', self.unreached_goals)
                        print('g', goal_state)

                        if np.round(goal_state,2) in self.unreached_goals:
                            index=np.where((self.unreached_goals ==np.round(goal_state,2)).all(axis=1))
                            self.unreached_goals = np.delete(self.unreached_goals, index,axis=0)
                            print('removed goal from unreached', np.round(goal_state,2))
                            print('unreached', self.unreached_goals)

                        if self.unreached==False:
                            self.proto_goals = np.delete(self.proto_goals, goal_idx,axis=0)
                            self.proto_goals_state = np.delete(self.proto_goals_state, goal_idx,axis=0)
                            self.proto_goals_dist = np.delete(self.proto_goals_dist, goal_idx,axis=0)
                        assert self.proto_goals.shape[0] == self.proto_goals_state.shape[0] == self.proto_goals_dist.shape[0]
                    else:
                        print('not implemented yet!!!!')

                    episode_reward=0
                    episode_step=0

                    self.unreached=False 
                    goal_idx = np.random.randint(self.proto_goals.shape[0])
                    goal_pix = self.proto_goals[goal_idx]
                    goal_state = self.proto_goals_state[goal_idx]

                    print('goal', goal_idx) 


                    if self.cfg.obs_type == 'pixels' and time_step1.last()==False:
                        print('last1')
                        self.replay_storage1.add_goal_general(time_step1, self.train_env1.physics.get_state(), meta, 
                                                              goal_pix, goal_state,
                                                      time_step_no_goal, True, last=True)


                    self.current_init = np.append(self.current_init, self.train_env1.physics.get_state()[None,:], axis=0)
                    print('current', self.current_init)
                    print('obs', self.train_env1.physics.get_state())
                    meta = self.agent.update_meta(meta, self._global_step, time_step1)

                    if self.pmm:
                        time_step, time_step_no_goal = self.make_env(actor1=False, init_idx=-1, goal_state=None, pmm=self.pmm)
                    else:
                        time_step = self.train_env.reset()   
                        print('before setting train env', self.train_env.physics.get_state())
                        print('before setting train env1', self.train_env1.physics.get_state())

                        with self.train_env.physics.reset_context():

                            self.train_env.physics.set_state(self.train_env1.physics.get_state())
                        act_ = np.zeros(self.train_env.action_spec().shape, self.train_env.action_spec().dtype)
                        time_step = self.train_env.step(act_) 
                        print('after setting train env', self.train_env.physics.get_state())
                        print('ts1', time_step1.observation['observations'])
                        print('ts', time_step.observation['observations'])

                if episode_step==499 and ((self.pmm==False and episode_reward < abs(self.cfg.reward_cutoff*20)) or (self.pmm and episode_reward<100)):
                    #keeping this for now so the unreached list doesn't grow too fast
                    if self.unreached==False:
                        self.unreached_goals=np.append(self.unreached_goals, self.proto_goals_state[goal_idx][None,:], axis=0)

                if self.global_step > (self.cfg.num_seed_frames):

                    metrics = self.agent.update(self.replay_iter1, self.global_step, actor1=True)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')


                episode_step += 1
                self._global_step += 1
            
            if self._global_step%200000==0 and self._global_step!=0:
                print('saving agent')
                path = os.path.join(self.work_dir, 'optimizer_{}_{}.pth'.format(str(self.cfg.agent.name),self._global_step))
                torch.save(self.agent, path)

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
    from pphg_gc_only import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
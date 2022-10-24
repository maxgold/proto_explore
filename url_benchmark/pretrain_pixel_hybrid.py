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
from scipy.spatial.distance import cdist
from logger import Logger, save
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_replay_buffer, ndim_grid, make_replay_offline
import matplotlib.pyplot as plt
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS


def make_agent(obs_type, obs_spec, action_spec, goal_shape,num_expl_steps, goal, cfg, hidden_dim, batch_size, update_gc, lr, offline=False, gc_only=False, intr_coef=0, switch_gc=False, film_gc=False):
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
    if film_gc:
        cfg.switch_gc = switch_gc
    if cfg.name=='proto_intr':
        cfg.intr_coef = intr_coef
    return hydra.utils.instantiate(cfg)

def get_state_embeddings(agent, states):
    with torch.no_grad():
        s = agent.encoder(states)
        s = agent.predictor(s)
        s = agent.projector(s)
        s = F.normalize(s, dim=1, p=2)
    return s


def heatmaps(self, env, model_step, replay_dir2, goal):
    if goal:
        replay_dir = self.work_dir / 'buffer1' / 'buffer_copy'
    else:
        replay_dir = self.work_dir / 'buffer2' / 'buffer_copy'
        
    replay_buffer = make_replay_offline(env,
                                Path(replay_dir),
                                2000000,
                                1,
                                0,
                                self.cfg.discount,
                                goal=goal,
                                relabel=False,
                                model_step=model_step,
                                replay_dir2=replay_dir2,
                                obs_type=self.cfg.obs_type, 
                                eval=True)
   
    if goal:
        states, actions, rewards, goal_state = replay_buffer.parse_dataset(goal_state=True)
    else:
        states, actions, rewards = replay_buffer.parse_dataset()
    
    #only adding states and rewards in replay_buffer
    if goal:
        tmp = np.hstack((states, goal_state, rewards))
        df = pd.DataFrame(tmp, columns= ['x', 'y', 'pos', 'v', 'g_x', 'g_y', 'gp', 'gv','r'])
    else:
        tmp = np.hstack((states, rewards))
        df = pd.DataFrame(tmp, columns= ['x', 'y', 'pos', 'v','r'])

    heatmap, _, _ = np.histogram2d(df.iloc[:, 0], df.iloc[:, 1], bins=50, 
                                   range=np.array(([-.29, .29],[-.29, .29])))
    plt.clf()
    
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
    ax.set_title(model_step)
    
    if goal:
        plt.savefig(f"./{model_step}_gc_heatmap.png")
        wandb.save(f"./{model_step}_gc_heatmap.png")
    else:
        plt.savefig(f"./{model_step}_proto_heatmap.png")
        wandb.save(f"./{model_step}_proto_heatmap.png")
    
    if goal:
        heatmap, _, _ = np.histogram2d(df.iloc[:, 4], df.iloc[:, 5], bins=50, 
                                   range=np.array(([-.29, .29],[-.29, .29])))
        plt.clf()
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        ax.set_title(model_step)
    
        plt.savefig(f"./{model_step}_gc_goals.png")
        wandb.save(f"./{model_step}_gc_goals.png")

    #percentage breakdown
    df=df*100
    heatmap, _, _ = np.histogram2d(df.iloc[:, 0], df.iloc[:, 1], bins=20, 
                                   range=np.array(([-29, 29],[-29, 29])))
    plt.clf()

    fig, ax = plt.subplots(figsize=(10,10))
    labels = np.round(heatmap.T/heatmap.sum()*100, 1)
    sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax, annot=labels).invert_yaxis()
    if goal:
        plt.savefig(f"./{self._global_step}_gc_heatmap_pct.png")
        wandb.save(f"./{self._global_step}_gc_heatmap_pct.png")
    else:
        plt.savefig(f"./{self._global_step}_proto_heatmap_pct.png")
        wandb.save(f"./{self._global_step}_proto_heatmap_pct.png")

    #rewards seen thus far
    df = df.astype(int)
    result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['r']
    result.fillna(0, inplace=True)
    plt.clf()
    fig, ax = plt.subplots()
    sns.heatmap(result, cmap="Blues_r", ax=ax).invert_yaxis()
    if goal:
        plt.savefig(f"./{self._global_step}_gc_reward.png")
        wandb.save(f"./{self._global_step}_gc_reward.png")
    else:
        plt.savefig(f"./{self._global_step}_proto_reward.png")
        wandb.save(f"./{self._global_step}_proto_reward.png")

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
        task = self.cfg.task
        self.no_goal_task = self.cfg.task_no_goal
        goal_array = ndim_grid(2,20)
        idx = np.random.randint(0,goal_array.shape[0])
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
        self.train_env2 = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                                  cfg.action_repeat, seed=None, goal=None)
        self.eval_env = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, seed=None, goal=self.first_goal)
        self.eval_env_no_goal = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                   cfg.action_repeat, seed=None, goal=None)
        self.eval_env_goal = dmc.make(self.no_goal_task, 'states', cfg.frame_stack,
                                   1, seed=None, goal=None)

        # create agent
        #import IPython as ipy; ipy.embed(colors='neutral')
        if self.cfg.film_gc:
            self.agent = make_agent(cfg.obs_type,
                                self.train_env2.observation_spec(),
                                self.train_env2.action_spec(),
                                (9, 84, 84),
                                cfg.num_seed_frames // cfg.action_repeat,
                                True,
                                cfg.agent,
                                cfg.hidden_dim,
                                cfg.batch_size,
                                cfg.update_gc,
                                cfg.lr,
                                cfg.offline,
                                False,
                                cfg.intr_coef,
                                cfg.switch_gc,
                                True)
        else:
            self.agent = make_agent(cfg.obs_type,
                                self.train_env2.observation_spec(),
                                self.train_env2.action_spec(),
                                (9, 84, 84),
                                cfg.num_seed_frames // cfg.action_repeat,
                                True,
                                cfg.agent,
                                cfg.hidden_dim,
                                cfg.batch_size,
                                cfg.update_gc,
                                cfg.lr,
                                cfg.offline,
                                False,
                                cfg.intr_coef) 
        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env2.observation_spec(),
                      self.train_env2.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage1 = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer1')
        self.replay_storage2 = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer2')
        

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
            self.replay_loader1 = make_replay_loader(self.replay_storage1,
                                                    False,
                                                    cfg.replay_buffer_gc,
                                                    cfg.batch_size_gc,
                                                    cfg.replay_buffer_num_workers,
                                                    False, cfg.nstep, cfg.discount,
                                                    True, cfg.hybrid_gc,cfg.obs_type,
                                                    cfg.hybrid_pct)

        #& then put them in two different eps_fns
        #if randomint > x, load episode in one of the eps_fns

        self.replay_loader2  = make_replay_loader(self.replay_storage2,
                                                False,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount,
                                                False, False,cfg.obs_type)
        
        
       # self.replay_buffer_intr = make_replay_buffer(self.eval_env,
       #                                                 self.work_dir / 'buffer2' / 'buffer_copy',
       #                                                 100000,
       #                                                 self.cfg.batch_size,
       #                                                 0,
       #                                                 self.cfg.discount,
       #                                                 goal=False,
       #                                                 relabel=False,
       #                                                 replay_dir2 = False,
       #                                                 )
        self._replay_iter1 = None
        self._replay_iter2 = None

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
    def replay_iter2(self):
        if self._replay_iter2 is None:
            self._replay_iter2 = iter(self.replay_loader2)
        return self._replay_iter2
    
    
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
    
#     def encoding_grid(self):
#         if self.loaded == False:
#             replay_dir = self.work_dir / 'buffer2' / 'buffer_copy'
#             self.replay_buffer_intr = make_replay_offline(self.eval_env,
#                                         replay_dir,
#                                         self.replay_buffer_gc,
#                                         self.cfg.batch_size,
#                                         0,
#                                         self.cfg.discount,
#                                         goal=False,
#                                         relabel=False,
#                                         model_step = self.global_step,
#                                         replay_dir2=False,
#                                         obs_type = self.cfg.obs_type
#                                         )
#             self.loaded = True
#             pix, states, actions = self.replay_buffer_intr._sample(eval_pixel=True)
#             pix = pix.astype(np.float64)
#             states = states.astype(np.float64)
#             states = states.reshape(-1,4)
#             grid = pix.reshape(9, 84, 84)
#             grid = torch.tensor(grid).cuda().float()
#             return grid, states
#         else:
#             pix, states, actions = self.replay_buffer_intr._sample(eval_pixel=True)
#             pix = pix.astype(np.float64)
#             states = states.astype(np.float64)
#             states = states.reshape(-1,4)
#             grid = pix.reshape(9, 84, 84)
#             grid = torch.tensor(grid).cuda().float()
#             return grid, states
        
        
        
        
    def encode_proto(self):
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
        else:
            
            replay_dir = self.work_dir / 'buffer2' / 'buffer_copy'
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
            
        state, actions, rewards = self.replay_buffer_intr.parse_dataset()
        idx = np.random.randint(0, state.shape[0], size=1024)
        state=state[idx]
        state=state.reshape(1024,4)
        print('state shape',state.shape)
        obs = torch.empty(1024, 9, 84, 84)
        states = torch.empty(1024, 4)
        grid_embeddings = torch.empty(1024, 128)
        meta = self.agent.init_meta()
        for i in range(1024):
            with torch.no_grad():
                with self.eval_env_goal.physics.reset_context():
                    time_step = self.eval_env_goal.physics.set_state(np.array([state[i][0], state[i][1], 0, 0]))
                time_step = self.eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
                goal = torch.tensor(np.transpose(time_step.copy(), (2,0,1))).cuda()
                obs[i] = torch.tile(goal, (1,3,1,1))
                states[i] = torch.tensor(state[i,:]).cuda().float()
#         import IPython as ipy; ipy.embed(colors='neutral')    
        obs = obs.cuda().float()
        grid_embeddings = get_state_embeddings(self.agent, obs)
        protos = self.agent.protos.weight.data.detach().clone()
        protos = F.normalize(protos, dim=1, p=2)
        dist_mat = torch.cdist(protos, grid_embeddings)
        closest_points = dist_mat.argmin(-1)
        proto2d = states[closest_points.cpu(), :2]
        return states

        
        

 #   def sample_goal_uniform(self, eval=False):
 #       if self.loaded_uniform == False:
 #           goal_index = pd.read_csv('/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/uniform_goal_pixel_index.csv')
 #           for ix in range(len(goal_index)):
 #               tmp = np.load('/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/exp_local/2022.08.27/224211_proto/buffer2/buffer_copy/'+goal_index.iloc[ix, 0])
 #               self.uniform_goal.append(np.array(tmp['observation'][int(goal_index.iloc[ix, -1])]))
 #               self.uniform_state.append(np.array(tmp['state'][int(goal_index.iloc[ix, -1])]))
 #           self.loaded_uniform = True
 #           self.count_uniform +=1
 #           print('loaded in uniform goals')
 #           return self.uniform_goal[self.count_uniform-1], self.uniform_state[self.count_uniform-1][:2]
 #       else:
 #           if self.count_uniform<400:
 #               self.count_uniform+=1
 #           else:
 #               self.count_uniform = 1
 #           return self.uniform_goal[self.count_uniform-1], self.uniform_state[self.count_uniform-1][:2]
    
    def sample_goal_distance(self, init_state=None):
        if self.goal_loaded==False and init_state is None:
            goal_array = ndim_grid(2,20)
            if init_state is None:
                dist_goal = cdist(np.array([[-.15,.15]]), goal_array, 'euclidean')
            else:
                dist_goal = cdist(np.array([[init_state[0],init_state[1]]]), goal_array, 'euclidean')
                
                
            df1 = pd.DataFrame()
            df1['distance'] = dist_goal.reshape((goal_array.shape[0],))
            df1['index'] = df1.index
            df1 = df1.sort_values(by='distance')
            goal_array_ = []
            for x in range(len(df1)):
                goal_array_.append(goal_array[df1.iloc[x,1]])
            self.distance_goal = goal_array_
            self.goal_loaded=True
            index=self.global_step//1000
            idx = np.random.randint(max(index-10,0),min(index+30, goal_array.shape[0]))

        
        elif self.goal_loaded and init_state is None:
            goal_array = ndim_grid(2,20)
            if self.global_step<500000:
                index=self.global_step//1000
                if index<goal_array.shape[0]:
                    idx = np.random.randint(max(index-10,0),min(index+30, goal_array.shape[0]))
                else:
                    idx = np.random.randint(0,goal_array.shape[0])
            else:
                idx = np.random.randint(0,goal_array.shape[0])

        else:
            goal_array = ndim_grid(2,20)
            dist_goal = cdist(np.array([[init_state[0],init_state[1]]]), goal_array, 'euclidean')

            df1 = pd.DataFrame()
            df1['distance'] = dist_goal.reshape((goal_array.shape[0],))
            df1['index'] = df1.index
            df1 = df1.sort_values(by='distance')
            goal_array_ = []
            for x in range(len(df1)):
                goal_array_.append(goal_array[df1.iloc[x,1]])
            self.distance_goal = goal_array_
            if self.global_step<500000:
                index=self.global_step//2000
                if index<goal_array.shape[0]:
                    idx = np.random.randint(max(index-10,0),min(index+30, goal_array.shape[0]))
                else:
                    idx = np.random.randint(0,goal_array.shape[0])
            else:
                idx = np.random.randint(0,goal_array.shape[0]) 		
        return self.distance_goal[idx]
    
        
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


    def eval(self):
        self.proto_goal = self.encode_proto()
        num = len(self.proto_goal)
#         idx = np.random.randint(0, num,size=(50,))
#         proto2d = proto2d[idx, :]
        plt.clf()
        fig, ax = plt.subplots()
        ax.scatter(self.proto_goal[:,0], self.proto_goal[:,1])
        plt.savefig(f"./prototypes_{self.global_step}.png")
        wandb.save(f"./prototypes_{self.global_step}.png")
        
        heatmaps(self, self.eval_env, self.global_step, False, True)
        heatmaps(self, self.eval_env, self.global_step, False, False)
        
        goal_array = ndim_grid(2,10)
        success=0
        df = pd.DataFrame(columns=['x','y','r'], dtype=np.float64)

        for ix, x in enumerate(goal_array):
            step, episode, total_reward = 0, 0, 0
         #   goal_pix, goal_state = self.sample_goal_uniform(eval=True)
            goal_state = np.array([x[0], x[1]])
            self.eval_env = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                    self.cfg.action_repeat, seed=None, goal=goal_state)
            #self.eval_env_no_goal = dmc.make(self.no_goal_task, self.cfg.obs_type, self.cfg.frame_stack,
             #       self.cfg.action_repeat, seed=None, goal=None)
            self.eval_env_goal = dmc.make(self.no_goal_task, 'states', self.cfg.frame_stack,
                    self.cfg.action_repeat, seed=None, goal=None)
            eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
            meta = self.agent.init_meta()

            while eval_until_episode(episode):
                time_step = self.eval_env.reset()
                print('time_step', time_step.observation['observations'])
                self.eval_env_no_goal = dmc.make(self.no_goal_task, self.cfg.obs_type, self.cfg.frame_stack,
                    self.cfg.action_repeat, seed=None, goal=goal_state, init_state=time_step.observation['observations'][:2])
                time_step_no_goal = self.eval_env_no_goal.reset()
                print('time_step no goal', time_step_no_goal.observation['observations'])
                #time_step_goal = self.eval_env_goal.reset()
                
                with self.eval_env_goal.physics.reset_context():
                    time_step_goal = self.eval_env_goal.physics.set_state(np.array([goal_state[0], goal_state[1], 0, 0]))
                time_step_goal = self.eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
                if ix%10==0:
                    self.video_recorder.init(self.eval_env, enabled=(episode == 0))
                
                while not time_step.last():
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
                    if ix%10==0:
                        self.video_recorder.record(self.eval_env)
                    total_reward += time_step.reward
                    step += 1
       
                episode += 1
                if ix%10==0:
                    self.video_recorder.save(f'{self.global_frame}_{ix}.mp4')
            
                if self.cfg.eval:
                    print('saving')
                    save(str(self.work_dir)+'/eval_{}.csv'.format(model.split('.')[-2].split('/')[-1]), [[x.cpu().detach().numpy(), total_reward, time_step.observation[:2], step]])

                else:
                    print('saving')
                    print(str(self.work_dir)+'/eval_{}.csv'.format(self._global_step))
                    save(str(self.work_dir)+'/eval_{}.csv'.format(self._global_step), [[goal_state, total_reward, time_step.observation['observations'], step]])
            if total_reward > 100*self.cfg.num_eval_episodes:
                success +=1
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
            

    def eval_intrinsic(self, model):
        obs = torch.empty(1024, 9, 84, 84)
        states = torch.empty(1024, 4)
        grid_embeddings = torch.empty(1024, 128)
        actions = torch.empty(1024,2)
        meta = self.agent.init_meta()
        for i in range(1024):
            with torch.no_grad():
                grid, state = self.encoding_grid()
                action = self.agent.act2(grid, meta, self._global_step, eval_mode=True)
                actions[i] = action
                obs[i] = grid
                states[i] = torch.tensor(state).cuda().float()
        import IPython as ipy; ipy.embed(colors='neutral')    
        obs = obs.cuda().float()
        actions = actions.cuda().float()
        grid_embeddings = get_state_embeddings(self.agent, obs)
        protos = self.agent.protos.weight.data.detach().clone()
        protos = F.normalize(protos, dim=1, p=2)
        dist_mat = torch.cdist(protos, grid_embeddings)
        closest_points = dist_mat.argmin(-1)
        proto2d = states[closest_points.cpu(), :2]
        with torch.no_grad():
            reward = self.agent.compute_intr_reward(obs, self._global_step)
            q_value = self.agent.get_q_value(obs, actions)
        for x in range(len(reward)):
            print('saving')
            print(str(self.work_dir)+'/eval_intr_reward_{}.csv'.format(self._global_step))
            save(str(self.work_dir)+'/eval_intr_reward_{}.csv'.format(self._global_step), [[obs[x].cpu().detach().numpy(), reward[x].cpu().detach().numpy(), q[x].cpu().detach().numpy(), self._global_step]])

        
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

        time_step2 = self.train_env2.reset()
        meta = self.agent.init_meta() 
         
        if self.cfg.obs_type == 'pixels':
            self.replay_storage1.add_goal(time_step1, meta, time_step_goal, time_step_no_goal,self.train_env_goal.physics.state(), True)
            print('replay1')
            self.replay_storage2.add(time_step2, meta, True)  
            print('replay2')
        else:
            self.replay_storage1.add_goal(time_step1, meta, goal)
            self.replay_storage2.add(time_step2, meta)  

        #self.train_video_recorder.init(time_step.observation)
        metrics = None
        goal_array = ndim_grid(2,20)
        while train_until_step(self.global_step):
            if (time_step1.last() and time_step2.last()) or episode_step==self.cfg.episode_length:
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
                        log('buffer_size', len(self.replay_storage2))
                        log('step', self.global_step)
                
                # reset env
               # time_step1 = self.train_env1.reset()
               # time_step2 = self.train_env2.reset()
               # time_step_no_goal = self.train_env_no_goal.reset()
                #time_step_goal = self.train_env_goal.reset()
               # meta = self.agent.init_meta()

                if self.cfg.obs_type =='pixels':
                    self.replay_storage1.add_goal(time_step1, meta,time_step_goal, time_step_no_goal, self.train_env_goal.physics.state(), True, last=True)
                    self.replay_storage2.add(time_step2, meta,True, last=True)
                #else:
                #    self.replay_storage.add(time_step, meta)

                #self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step) and self.global_step!=0:
                #print('trying to evaluate')
                self.eval()
                #plt.clf()
                #fig, ax = plt.subplots()
                #sr, idx = zip(*sorted(zip(self.global_success_rate, self.global_index)))

                #ax.ticklabel_format(style='plain')
                #ax.plot(idx,sr)
                #plt.savefig(f"./{self._global_step}_eval.png")
                #wandb.save("./{self._global_step}_eval.png")
                    #self.eval_intrinsic(model)
                #else:
                    #self.logger.log('eval_total_time', self.timer.total_time(),
                    #            self.global_frame)
                    #self.eval()
                    #self.logger.log('eval_total_time', self.timer.total_time(),
                    #    self.global_frame)
                    
           # meta = self.agent.update_meta(meta, self._global_step, time_step1)
            
            if episode_step== 0 and self.global_step!=0:
                
                if self.cfg.curriculum:
                    if self.cfg.const_init==False and self.global_step%5000!=0:
                        init_state = time_step1.observation['observations'][:2]
                        print('init', init_state)
                        goal_=self.sample_goal_distance(init_state)
                        goal_state = np.array([goal_[0], goal_[1]])
                    else:
                        goal_=self.sample_goal_distance()
                        goal_state = np.array([goal_[0], goal_[1]])
                    
                elif self.cfg.sample_proto:
                    if (self.cfg.num_seed_frames==self.global_step) or (self.global_step < 100000 and self.global_step%2000==0) or (300000>self.global_step >100000 and self.global_step%50000==0):
                        self.proto_goal = self.encode_proto()
                        print('proto_goal', self.proto_goal)
                        idx = np.random.randint(0, self.proto_goal.shape[0])
                        goal_state = np.array([self.proto_goal[idx][0], self.proto_goal[idx][1]])
                    elif len(self.proto_goal)>0:
                        idx = np.random.randint(0, self.proto_goal.shape[0])
                        goal_state = np.array([self.proto_goal[idx][0], self.proto_goal[idx][1]])
                    else:
                        print('havent sampled prototypes yet, sampling randomly')
                        goal_array = ndim_grid(2,20)
                        idx = np.random.randint(0,len(goal_array))
                        goal_state = np.array([goal_array[idx][0], goal_array[idx][1]])

                    
                else:
                    if self.global_step<100000:
                        init_state = np.random.uniform(.02,.29,(2,))
                        goal_state = np.array([-init_state[0], init_state[1]])
                    else:
                        goal_array = ndim_grid(2,20)
                        idx = np.random.randint(0, goal_array.shape[0])
                        goal_state = np.array([goal_array[idx][0], goal_array[idx][1]])

                if self.cfg.const_init==False and self.global_step%5000!=0:
                    self.train_env1 = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                                                  self.cfg.action_repeat, seed=None, goal=goal_state, init_state=init_state)
                else:
                    self.train_env1 = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                                                  self.cfg.action_repeat, seed=None, goal=goal_state)
                time_step1 = self.train_env1.reset()
                self.train_env_no_goal = dmc.make(self.no_goal_task, self.cfg.obs_type, self.cfg.frame_stack,
                self.cfg.action_repeat, seed=None, goal=goal_state, init_state=time_step1.observation['observations'][:2])
                time_step_no_goal = self.train_env_no_goal.reset()
                time_step2 = self.train_env2.reset()
                #time_step_goal = self.train_env_goal.reset()
                meta = self.agent.update_meta(meta, self._global_step, time_step1) 
                print('sampled goal', goal_state)

                with self.train_env_goal.physics.reset_context():
                    time_step_goal = self.train_env_goal.physics.set_state(np.array([goal_state[0], goal_state[1],0,0]))
                
                time_step_goal = self.train_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
               
                if self.cfg.obs_type == 'pixels' and time_step1.last()==False:
                    self.replay_storage1.add_goal(time_step1, meta,time_step_goal, time_step_no_goal,self.train_env_goal.physics.state(), True)
                    self.replay_storage2.add(time_step2,meta,True)
            
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                if self.cfg.obs_type == 'pixels':

                    action1 = self.agent.act(time_step_no_goal.observation['pixels'].copy(),
                                            time_step_goal.copy(),
                                            meta,
                                            self._global_step,
                                            eval_mode=False)

                    action2 = self.agent.act2(time_step2.observation['pixels'].copy(),
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
            time_step2 = self.train_env2.step(action2)
            episode_reward += time_step1.reward
            
            
            if self.cfg.obs_type == 'pixels' and time_step1.last()==False and episode_step!=self.cfg.episode_length:
                self.replay_storage1.add_goal(time_step1, meta, time_step_goal, time_step_no_goal,self.train_env_goal.physics.state(), True)
                self.replay_storage2.add(time_step2, meta, True)
            elif (self.cfg.obs_type == 'pixels' and time_step1.last()) or (self.cfg.obs_type == 'pixels' and episode_step==self.cfg.episode_length):
                self.replay_storage1.add_goal(time_step1, meta, time_step_goal, time_step_no_goal,self.train_env_goal.physics.state(), True, last=True)
                self.replay_storage2.add(time_step2, meta, True)
            elif self.cfg.obs_type == 'states':
                self.replay_storage1.add_goal(time_step1, meta, goal)
                self.replay_storage2.add(time_step2, meta)
            
            
            episode_step += 1
            
            if episode_reward > 100 and self.cfg.resample_goal and time_step1.last()==False and episode_step<(self.cfg.episode_length-50):
                print('reached making new env')
                episode_reward=0
                current_state = time_step1.observation['observations'][:2]

                self.replay_storage1.add_goal(time_step1, meta,time_step_goal, time_step_no_goal,self.train_env_goal.physics.state(), True, last=True)
                self.replay_storage2.add(time_step2,meta,True, last=True)

                if self.cfg.curriculum:
                    goal_=self.sample_goal_distance(current_state)
                    goal_state = np.array([goal_[0], goal_[1]])
                else:
                    goal_array = ndim_grid(2,20)
                    idx = np.random.randint(0, goal_array.shape[0])
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
                time_step2 = self.train_env2.reset()
                #time_step_goal = self.train_env_goal.reset()
                meta = self.agent.update_meta(meta, self._global_step, time_step1)
                print('sampled goal', goal_state)

                with self.train_env_goal.physics.reset_context():
                    time_step_goal = self.train_env_goal.physics.set_state(np.array([goal_state[0], goal_state[1],0,0]))

                time_step_goal = self.train_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))

                if self.cfg.obs_type == 'pixels' and time_step1.last()==False and episode_step!=self.cfg.episode_length:
                    self.replay_storage1.add_goal(time_step1, meta,time_step_goal, time_step_no_goal,self.train_env_goal.physics.state(), True)
                    self.replay_storage2.add(time_step2,meta,True)
                elif (self.cfg.obs_type == 'pixels' and time_step1.last()) or (self.cfg.obs_type == 'pixels' and episode_step==self.cfg.episode_length):
                    self.replay_storage1.add_goal(time_step1, meta,time_step_goal, time_step_no_goal,self.train_env_goal.physics.state(), True, last=True)
                    self.replay_storage2.add(time_step2,meta,True)

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
                metrics = self.agent.update(self.replay_iter2, self.global_step)
            
            self._global_step += 1

            if self._global_step%50000==0 and self._global_step!=0:
                print('saving agent')
                path = os.path.join(self.work_dir, 'optimizer_{}_{}.pth'.format(str(self.cfg.agent.name),self._global_step))
                torch.save(self.agent, path)
                path_2 = os.path.join(self.work_dir, 'encoder_{}_{}.pth'.format(str(self.cfg.agent.name),self._global_step))
                torch.save(self.agent.encoder, path_2)

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
    from pretrain_pixel_hybrid import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()

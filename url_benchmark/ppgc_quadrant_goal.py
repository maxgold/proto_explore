import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
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


def make_agent(obs_type, obs_spec, action_spec, goal_shape,num_expl_steps, goal, cfg, hidden_dim, batch_size, update_gc, lr, offline=False, gc_only=False, intr_coef=0):
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


def heatmaps(self, env, model_step, replay_dir2, goal):
    if self.cfg.offline:
        replay_dir = self.replay_goal_dir 
    elif goal:
        replay_dir = self.work_dir / 'buffer1' / 'buffer_copy'
    else:
        replay_dir = self.work_dir / 'buffer2' / 'buffer_copy'
    print('heatmap buffer')
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

    states, actions, rewards = replay_buffer.parse_dataset()
    #only adding states and rewards in replay_buffer
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
 
        # create agent
        #import IPython as ipy; ipy.embed(colors='neutral')
        self.agent = make_agent(cfg.obs_type,
                                self.train_env1.observation_spec(),
                                self.train_env1.action_spec(),
                                (9, 84, 84),
                                cfg.num_seed_frames // cfg.action_repeat,
                                True,
                                cfg.agent,
                                cfg.hidden_dim, 
                                cfg.batch_size_gc, 
                                cfg.update_gc, 
                                cfg.lr, 
                                cfg.offline, 
                                True, 
                                cfg.intr_coef)
        
        encoder = torch.load('/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/encoder/2022.08.28/222511_proto1/encoder_proto1_900000.pth')
        self.agent.init_encoder_from(encoder)
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
        self.replay_goal_dir = Path('/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/exp_local/2022.09.04/022144_proto/buffer2/buffer_copy/') 

        # create replay buffer
        if cfg.offline:
            print('make buffer1')
            self.replay_loader1 = make_replay_buffer(self.eval_env,
                                                    self.replay_goal_dir,
                                                    cfg.replay_buffer_size,
                                                    cfg.batch_size_gc,
                                                    cfg.replay_buffer_num_workers,
                                                    self.cfg.discount,
                                                    goal=True,
                                                    relabel=False,
                                                    replay_dir2=False,
                                                    obs_type = cfg.obs_type,
                                                    model_step=1000,
                                                    offline=cfg.offline,
                                                    nstep=cfg.nstep,
                                                    load_every=cfg.batch_size_gc*100)
        elif cfg.hybrid:
            print('make buffer hybrid')
            self.replay_loader1 = make_replay_buffer(self.eval_env,
                                                    self.work_dir / "buffer1",
                                                    cfg.replay_buffer_size,
                                                    cfg.batch_size_gc,
                                                    cfg.replay_buffer_num_workers,
                                                    self.cfg.discount,
                                                    goal=True,
                                                    relabel=False,
                                                    replay_dir2=self.replay_goal_dir,
                                                    obs_type = cfg.obs_type,
                                                    model_step=3000,
                                                    nstep=cfg.nstep,
                                                    hybrid=True,
                                                    hybrid_pct=self.cfg.hybrid_pct)

            #self.replay_loader1 = make_replay_loader(self.replay_storage1,
            #                                        False,
            #                                        cfg.replay_buffer_gc,
            #                                        cfg.batch_size_gc,
            #                                        cfg.replay_buffer_num_workers,
            #                                        False, cfg.nstep, cfg.discount,
            #                                         True, cfg.hybrid,cfg.obs_type, 
            #                                         cfg.hybrid_pct, actor1=True,
            #                                         replay_dir2=self.replay_goal_dir, model_step=1000000)
             

        else:

            self.replay_loader1 = make_replay_loader(self.replay_storage1,
                                                False,
                                                cfg.replay_buffer_gc,
                                                cfg.batch_size_gc,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount,
                                                True, cfg.hybrid,cfg.obs_type, cfg.hybrid_pct, actor1=True)
    #    self.replay_buffer_intr = make_replay_buffer(self.eval_env,
    #                                                    self.work_dir / 'buffer2' / 'buffer_copy',
    #                                                    100000,
    #                                                    1,
    #                                                    0,
    #                                                    self.cfg.discount,
    #                                                    goal=False,
    #                                                    relabel=False,
    #                                                    replay_dir2 = False,
    #                                                    )
        # self.replay_loader2  = make_replay_loader(self.replay_storage2,
       #                                         cfg.replay_buffer_size,
       #                                         cfg.batch_size,
       #                                         cfg.replay_buffer_num_workers,
       #                                         False, cfg.nstep, cfg.discount,
       #                                         False, cfg.obs_type)
        self._replay_iter1 = None
       # self._replay_iter2 = None

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

  #  @property
  #  def replay_iter2(self):
  #      if self._replay_iter2 is None:
  #          self._replay_iter2 = iter(self.replay_loader2)
  #      return self._replay_iter2
    
    
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
    
    def encoding_grid():
        if self.loaded == False:
            replay_dir = self.work_dir / 'buffer2' / 'buffer_copy'
            print('make encoding grid buffer 2')
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
            self.loaded = True
            pix, states, actions = self.replay_buffer_intr._sample(eval_pixel=True)
            pix = pix.astype(np.float64)
            states = states.astype(np.float64)
            states = states.reshape(-1,2)
            grid = pix.reshape(-1,9, 84, 84)
            grid = torch.tensor(grid).cuda().float()
            return grid, states
        else:
            pix, states, actions = self.replay_buffer_intr._sample(eval_pixel=True)
            pix = pix.astype(np.float64)
            states = states.astype(np.float64)
            states = states.reshape(-1,2)
            grid = pix.reshape(-1,9, 84, 84)
            grid = torch.tensor(grid).cuda().float()
            return grid, states

    def sample_goal_distance(self):

        if self.goal_loaded==False:

            goal_array = ndim_grid(2,20)
            dist_goal = cdist(np.array([[-.15,.15]]), goal_array, 'euclidean')  
            df1=pd.DataFrame()
            df1['distance'] = dist_goal.reshape((400,))
            df1['index'] = df1.index
            df1 = df1.sort_values(by='distance')
            goal_array_ = []
            for x in range(len(df1)):
                if self.cfg.bottom_left:
                    if goal_array[df1.iloc[x,1]][0]<0 and goal_array[df1.iloc[x,1]][1]<0:
                        goal_array_.append(goal_array[df1.iloc[x,1]])
                else:
                    goal_array_.append(goal_array[df1.iloc[x,1]])
            self.distance_goal = goal_array_
            self.goal_loaded=True
            index=self.global_step//20000
            idx = np.random.randint(index,min(index+20, len(goal_array_)))

        else:
            if self.global_step<200000:
                index=self.global_step//2000
                if index < len(self.distance_goal):
                    idx = np.random.randint(index,min(index+20, len(self.distance_goal)))
                else:
                    idx = np.random.randint(0, len(self.distance_goal))
            else:
                idx = np.random.randint(0, len(self.distance_goal))
        return self.distance_goal[idx]




    def eval_goal(self):

        for i in range(10):
            step, episode, total_reward = 0, 0, 0
            goal_pix, goal_state = self.sample_goal_pixel(eval=True)
            self.eval_env = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                                                  self.cfg.action_repeat, seed=None, goal=goal_state, 
                                                  )
            eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
            meta = self.agent.init_meta()
            while eval_until_episode(episode):
                time_step = self.eval_env.reset()
              #  self.video_recorder.init(self.eval_env, enabled=(episode == 0))
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        if self.cfg.goal:
                            action = self.agent.act(time_step.observation['pixels'],
                                                    goal_pix,
                                                    meta,
                                                    self._global_step,
                                                    eval_mode=True)
                        else:
                            action = self.agent.act(time_step.observation,
                                                    meta,
                                                    self._global_step,
                                                    eval_mode=True)
                    time_step = self.eval_env.step(action)
                   # self.video_recorder.record(self.eval_env)
                    total_reward += time_step.reward
                    step += 1
       
                episode += 1
               # self.video_recorder.save(f'{self.global_frame}.mp4')
            
                if self.cfg.eval:
                    print('saving')
                    save(str(self.work_dir)+'/eval_{}.csv'.format(model.split('.')[-2].split('/')[-1]), [[x.cpu().detach().numpy(), total_reward, time_step.observation[:2], step]])

                else:
                    print('saving')
                    print(str(self.work_dir)+'/eval_{}.csv'.format(self._global_step))
                    save(str(self.work_dir)+'/eval_{}.csv'.format(self._global_step), [[goal_state, total_reward, time_step.observation['observations'][2:], step]])
        
            #if total_reward < 500*self.cfg.num_eval_episodes:
            #    self.unreachable.append([goal_pix, goal_state])
    

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
        heatmaps(self, self.eval_env, self.global_step, False, True)

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
        self.train_env_no_goal = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
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
        metrics = None
        
        while train_until_step(self.global_step):

            if self.cfg.offline:
                if eval_every_step(self.global_step) and self.global_step!=0:
                    self.eval()

                metrics = self.agent.update(self.replay_iter1, self.global_step, actor1=True)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
                
                if self.global_step%500==0:
                    elapsed_time, total_time = self.timer.reset()
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                          ty='train') as log:

                        log('fps', 1000 / elapsed_time)
                        log("total_time", total_time)
                        log("step", self.global_step)
                self._global_step += 1
                
            else:
                
                if time_step1.last():
                    print('last')
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
                            log('buffer_size', len(self.replay_storage1))
                            log('step', self.global_step)

                    #meta = self.agent.init_meta()
                    if self.cfg.obs_type =='pixels':
                        self.replay_storage1.add_goal(time_step1, meta,time_step_goal, time_step_no_goal, self.train_env_goal.physics.state(), True)
                    else:
                        self.replay_storage.add(time_step, meta)
                    self.train_video_recorder.init(self.train_env_goal)
                    # try to save snapshot
                    if self.global_frame in self.cfg.snapshots:
                        self.save_snapshot()
                    episode_step = 0
                    episode_reward = 0
			  
                # try to evaluate
                if eval_every_step(self.global_step) and self.global_step!=0:
                    self.eval()
                    plt.clf()
                    fig, ax = plt.subplots()
                    sr, idx = zip(*sorted(zip(self.global_success_rate, self.global_index)))

                    ax.ticklabel_format(style='plain')
                    ax.plot(idx,sr)
                    plt.savefig(f"./{self._global_step}_eval.png")
                    wandb.save("./{self._global_step}_eval.png")


                #meta = self.agent.update_meta(meta, self._global_step, time_step1)

                if episode_step  == 0 and self.global_step!=0:

                #if seed_until_step(self._global_step):
                    #if self.cfg.uniform:
                    #    goal_pix, goal_state = self.sample_goal_uniform()
                    #else:
                    #    goal_pix, goal_state = self.sample_goal_pixel()
                #print('sampled goal', goal_state)
                    if self.cfg.curriculum:
                        goal_=self.sample_goal_distance()
                        goal_state = np.array([goal_[0], goal_[1]])
                    
                    else:
                        idx = np.random.randint(0,400)
                        goal_array = ndim_grid(2,20)
                        goal_state = np.array([goal_array[idx][0], goal_array[idx][1]])
                    self.train_env1 = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                                                  self.cfg.action_repeat, seed=None, goal=goal_state)

                    time_step1 = self.train_env1.reset()
                    self.train_env_no_goal = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                        self.cfg.action_repeat, seed=None, goal=goal_state, init_state=time_step1.observation['observations'][:2])
                    time_step_no_goal = self.train_env_no_goal.reset()
                    meta = self.agent.update_meta(meta, self._global_step, time_step1)
                    print('sampled goal', goal_state)
                    
                    with self.train_env_goal.physics.reset_context():
                        time_step_goal = self.train_env_goal.physics.set_state(np.array([goal_state[0], goal_state[1],0,0]))
                    time_step_goal = self.train_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
                    
                    if self.cfg.obs_type == 'pixels' and time_step1.last()==False:
                        self.replay_storage1.add_goal(time_step1, meta, time_step_goal, time_step_no_goal,self.train_env_goal.physics.state(), True)
                    
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
                
                elif self.cfg.obs_type =='states':
                    self.replay_storage1.add_goal(time_step1, meta, goal)
                    self.replay_storage2.add(time_step2, meta)

                episode_step += 1

                if not seed_until_step(self.global_step):
                    metrics = self.agent.update(self.replay_iter1, self.global_step, actor1=True)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

                self._global_step += 1
 
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
    from ppgc_quadrant_goal import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()

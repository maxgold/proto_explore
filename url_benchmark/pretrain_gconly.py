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

def make_agent(self,obs_type, obs_spec, action_spec, goal_shape,num_expl_steps, goal, cfg, hidden_dim, batch_size, lr, feature_dim=50):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    cfg.goal_shape = goal_shape
    cfg.goal = goal
    cfg.hidden_dim = hidden_dim
    cfg.batch_size = batch_size
    cfg.lr = lr
    cfg.feature_dim=feature_dim
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
                                cfg.lr,
                                feature_dim=cfg.feature_dim)
 
        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env1.observation_spec(),
                      self.train_env1.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage1 = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage1,
                                                False,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount,
                                                goal=cfg.goal,
                                                obs_type=cfg.obs_type,
                                                hybrid=cfg.hybrid_gc,
                                                hybrid_pct=cfg.hybrid_pct,
                                                asym=cfg.asym,
                                                sl=cfg.sl)
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
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

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
                            if self.cfg.asym:
                                action = self.agent.act(time_step_no_goal.observation['pixels'],
                                                        np.array([goal_state[0], goal_state[1], 0, 0]),
                                                        meta,
                                                        self._global_step,
                                                        eval_mode=True)
                            else:
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
        if self.cfg.asym==False:
            time_step_goal = self.train_env_goal.reset()
            with self.train_env_goal.physics.reset_context():
                time_step_goal = self.train_env_goal.physics.set_state(np.array([self.first_goal[0], self.first_goal[1], 0, 0]))
        
            time_step_goal = self.train_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))

        meta = self.agent.init_meta() 
         
        if self.cfg.obs_type == 'pixels' and self.cfg.asym==False:
            self.replay_storage1.add_goal(time_step1, meta, time_step_goal, time_step_no_goal,self.train_env_goal.physics.state(), pixels=True)
            print('replay1')

        elif self.cfg.asym :
            self.replay_storage1.add_goal(time_step1, meta, np.array([self.first_goal[0], self.first_goal[1],0,0]), time_step_no_goal, goal_state=self.first_goal, pixels=True, asym=True)

        #self.train_video_recorder.init(time_step.observation)
        metrics = None 

        goal_state = self.first_goal
        goal_array = ndim_grid(2,20)
        while train_until_step(self.global_step):
            if time_step1.last():
                self._global_episode += 1
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

                if self.cfg.obs_type == 'pixels' and self.cfg.asym==False:
                    self.replay_storage1.add_goal(time_step1, meta,time_step_goal, time_step_no_goal, self.train_env_goal.physics.state(), True, last=True)
                elif self.cfg.obs_type == 'pixels' and self.cfg.asym:
                    self.replay_storage1.add_goal(time_step1, meta,time_step_goal, time_step_no_goal, goal_state=goal_state, pixels=True, last=True)
                else: 
                    self.replay_storage.add(time_step1, meta)
                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

                # reset env
                if self.cfg.const_init==False:
                    task = PRIMAL_TASKS[self.cfg.domain]
                    rand_init = np.random.uniform(.02,.29,size=(2,))
                    sign = np.array([[1,1],[-1,1],[1,-1],[-1,-1]])
                    rand = np.random.randint(4)
                    self.train_env1 = dmc.make(task, self.cfg.obs_type, self.cfg.frame_stack,
                                                              self.cfg.action_repeat, self.cfg.seed, init_state=(rand_init[0]*sign[rand][0], rand_init[1]*sign[rand][1]))
                    print('sampled init', (rand_init[0]*sign[rand][0], rand_init[1]*sign[rand][1]))
                time_step1 = self.train_env1.reset()
                meta = self.agent.init_meta() 

            # try to evaluate
            if eval_every_step(self.global_step) and self.global_step!=0:
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                self.eval()
            if (episode_step== 0 and self.global_step!=0):
                if self.cfg.curriculum:
                    goal_array = ndim_grid(2,20)
                    if self.global_step<100000:
                        init_state = np.random.uniform(.02,.29,(2,))
                        goal_state = np.array([-init_state[0], init_state[1]])
                    else:
                        goal_array = ndim_grid(2,20)
                        idx = np.random.randint(0, goal_array.shape[0])
                        goal_state = np.array([goal_array[idx][0], goal_array[idx][1]]) 
                else:
                    goal_array = ndim_grid(2,20)
                    idx = self.count
                    print('count', self.count)
                    self.count += 1
                    if self.count == len(goal_array):
                        self.count = 0
                    goal_state = np.array([goal_array[idx][0], goal_array[idx][1]])
                meta = self.agent.update_meta(meta, self.global_step, time_step1)
            
                self.train_env1 = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                                                                                      self.cfg.action_repeat, self.cfg.seed, goal=goal_state)
                time_step1 = self.train_env1.reset()
                self.train_env_no_goal = dmc.make(self.no_goal_task, self.cfg.obs_type, self.cfg.frame_stack,
                self.cfg.action_repeat, seed=None, goal=goal_state, init_state=time_step1.observation['observations'][:2])
                time_step_no_goal = self.train_env_no_goal.reset()
                meta = self.agent.update_meta(meta, self._global_step, time_step1) 
                print('time step', time_step1.observation['observations'])
                print('sampled goal', goal_state)
                
                if self.cfg.asym==False:
                    with self.train_env_goal.physics.reset_context():
                        time_step_goal = self.train_env_goal.physics.set_state(np.array([goal_state[0], goal_state[1],0,0]))

                    time_step_goal = self.train_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))

                if self.cfg.obs_type == 'pixels' and time_step1.last()==False and self.cfg.asym==False:
                    self.replay_storage1.add_goal(time_step1, meta,time_step_goal, time_step_no_goal,self.train_env_goal.physics.state(), True)  
                elif self.cfg.obs_type == 'pixels' and time_step1.last()==False and self.cfg.asym:
                    self.replay_storage1.add_goal(time_step1, meta, np.array([goal_state[0], goal_state[1],0,0]), time_step_no_goal, goal_state, True, asym=True)
            
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                if self.cfg.goal:
                    if self.cfg.obs_type=='pixels' and self.cfg.asym==False:
                        action = self.agent.act(time_step_no_goal.observation['pixels'].copy(),
                                                time_step_goal.copy(),
                                                meta,
                                                self._global_step,
                                                eval_mode=False) 
                    elif self.cfg.obs_type=='pixels' and self.cfg.asym:    
                        action = self.agent.act(time_step_no_goal.observation['pixels'].copy(),
                                            np.array([goal_state[0], goal_state[1], 0, 0]),
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
                metrics = self.agent.update(self.replay_iter, self.global_step, actor1=True)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
            
            #save agent
            if self._global_step%100000==0 and self.global_step!=0:
                path = os.path.join(self.work_dir, 'optimizer_{}_{}.pth'.format(str(self.cfg.agent.name),self._global_step))
                torch.save(self.agent, path)
            # take env step
            time_step1 = self.train_env1.step(action)
            time_step_no_goal = self.train_env_no_goal.step(action)
            episode_reward += time_step1.reward

            if self.cfg.obs_type == 'pixels' and time_step1.last()==False and self.cfg.asym==False:
                self.replay_storage1.add_goal(time_step1, meta, time_step_goal, time_step_no_goal,self.train_env_goal.physics.state(), pixels=True)
            elif self.cfg.obs_type == 'pixels' and time_step1.last()==False and self.cfg.asym:
                self.replay_storage1.add_goal(time_step1, meta, np.array([goal_state[0], goal_state[1],0,0]), time_step_no_goal,goal_state, pixels=True, asym=True)
            elif self.cfg.obs_type == 'pixels' and time_step1.last() and self.cfg.asym:
                
                self.replay_storage1.add_goal(time_step1, meta, np.array([goal_state[0], goal_state[1],0,0]), time_step_no_goal,goal_state, pixels=True, asym=True, last=True)
            elif self.cfg.obs_type == 'states':
                self.replay_storage1.add_goal(time_step1, meta, goal)

            

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
    from pretrain_gconly import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()

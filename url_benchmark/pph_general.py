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
#from eval_utils import heatmaps, eval_proto, eval_general, eval, eval_intrinsic

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS


def make_agent(obs_type, obs_spec, action_spec, goal_shape, num_expl_steps, cfg, lr=.0001, hidden_dim=1024, num_protos=512, update_gc=2, gc_only=False, offline=False, tau=.1, num_iterations=3, feature_dim=50, pred_dim=128, proj_dim=512, batch_size=1024, update_proto_every=10, lagr=.2, margin=.5, lagr1=.2, lagr2=.2, lagr3=.3, stddev_schedule=.2, stddev_clip=.3, update_proto=2, stddev_schedule2=.2, stddev_clip2=.3):

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
    print('shape', obs_spec.shape)
    return hydra.utils.instantiate(cfg)



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


        heatmap_pct = self.replay_storage1.state_visitation_gc_pct

        plt.clf()
        fig, ax = plt.subplots(figsize=(10,10))
        labels = np.round(heatmap_pct.T/heatmap_pct.sum()*100, 1)
        sns.heatmap(np.log(1 + heatmap_pct.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        ax.set_title(model_step)

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
        
        #moving avg of pixels != 0 :
        

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
            self.train_env_goal = dmc.make(self.no_goal_task, 'states', cfg.frame_stack,
                                       1, seed=None, goal=None)
            self.train_env = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                                      cfg.action_repeat, seed=None)
            self.eval_env = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                     cfg.action_repeat, seed=None, goal=self.first_goal)
            self.eval_env_no_goal = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                       cfg.action_repeat, seed=None, goal=None)
            self.eval_env_goal = dmc.make(self.no_goal_task, 'states', cfg.frame_stack,
                                       1, seed=None, goal=None)
        else:
            self.train_env1 = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                       cfg.action_repeat, seed=None)
            self.train_env_goal = dmc.make(self.cfg.task, 'states', cfg.frame_stack,
                                       1, seed=None, goal=None)
            self.train_env = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                                      cfg.action_repeat, seed=None)
            self.eval_env = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                     cfg.action_repeat, seed=None)
            



        # create agent

        if self.cfg.agent.name=='protox':
            self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                (3,84,84),
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
                                lagr1=cfg.lagr1,
                                lagr2=cfg.lagr2,
                                lagr3=cfg.lagr3,
                                margin=cfg.margin,
                                update_proto_every=cfg.update_proto_every,
                                stddev_schedule=cfg.stddev_schedule, 
                                stddev_clip=cfg.stddev_clip,
                                stddev_schedule2=cfg.stddev_schedule2,
                                stddev_clip2=cfg.stddev_clip2
                                )
        else: 
            self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                (3,84,84),
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
                                stddev_clip2=cfg.stddev_clip2)
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
        

        # create replay buffer
        print('regular or hybrid_gc loader')
        if self.cfg.combine_storage_gc:
            self.replay_loader1 = make_replay_loader(self.replay_storage1,
                                                    True,
                                                    cfg.replay_buffer_gc,
                                                    cfg.batch_size_gc,
                                                    cfg.replay_buffer_num_workers,
                                                    False, cfg.nstep, cfg.discount,
                                                    True, cfg.hybrid_gc,cfg.obs_type,
                                                    cfg.hybrid_pct, replay_dir2=self.work_dir / 'buffer2',
                                                    loss=cfg.loss_gc, test=cfg.test,
                                                    tile=cfg.frame_stack,
                                                    pmm=self.pmm,
                                                    obs_shape=self.train_env1.physics.state().shape[0])
        else:
            self.replay_loader1 = make_replay_loader(self.replay_storage1,
                                                    False,
                                                    cfg.replay_buffer_gc,
                                                    cfg.batch_size_gc,
                                                    cfg.replay_buffer_num_workers,
                                                    False, cfg.nstep, cfg.discount,
                                                    True, cfg.hybrid_gc,cfg.obs_type,
                                                    cfg.hybrid_pct, loss=cfg.loss_gc, test=cfg.test,
                                                    tile=cfg.frame_stack,
                                                    pmm=self.pmm,
                                                    obs_shape=self.train_env1.physics.state().shape[0])

        self.replay_loader = make_replay_loader(self.replay_storage,
                                                False,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount,
                                                goal=False,
                                                obs_type=cfg.obs_type,
                                                loss=cfg.loss,
                                                test=cfg.test) 

        self._replay_iter = None
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
        self.distance_goal_init = {}
        self.proto_goals_dist = np.zeros((10, 1))
        self.proto_goals = np.zeros((10, 2))
        self.proto_goals_matrix = np.zeros((60,60))
        self.proto_goals_id = np.zeros((10, 2))
        self.actor=True
        self.actor1=False
        self.final_df = pd.DataFrame(columns=['avg', 'med', 'max', 'q7', 'q8', 'q9'])
        self.reached_goals=np.empty((0,2))
        self.proto_goals_alt=[]
        self.proto_explore=False
        self.proto_explore_count=0
        self.gc_explore=False
        self.gc_explore_count=0
        self.goal_freq = np.zeros((6,6))
        self.previous_matrix = None
        self.current_matrix = None
        self.v_queue_ptr = 0 
        self.v_queue = np.zeros((2000,))
        self.count=0
        self.mov_avg_5 = np.zeros((2000,))
        self.mov_avg_10 = np.zeros((2000,))
        self.mov_avg_20 = np.zeros((2000,))
        self.mov_avg_50 = np.zeros((2000,))
        self.mov_avg_100 = np.zeros((2000,))
        self.mov_avg_200 = np.zeros((2000,))
        self.mov_avg_500 = np.zeros((2000,))
        self.unreached_goals = np.empty((0,2))
    
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
    
    def eval_proto(self):
        
        if self.pmm:
            heatmaps(self, self.eval_env, self.global_step, False, True, model_step_lb=False,gc=True,proto=True)
            eval_env_goal = dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None)
        
        ########################################################################
        #how should we measure exploration in non-pmm w/o heatmaps

        
        protos = self.agent.protos.weight.data.detach().clone()

        
        #####################################################################
        #probably need to cut down .5mil in buffer for non-pmm
        replay_buffer = make_replay_offline(self.eval_env,
                                                self.work_dir / 'buffer2' / 'buffer_copy',
                                                500000,
                                                0,
                                                0,
                                                .99,
                                                goal=False,
                                                relabel=False,
                                                model_step = self._global_step,
                                                replay_dir2=False,
                                                obs_type = 'pixels'
                                                )

        
        state, actions, rewards, eps, index = replay_buffer.parse_dataset() 
        state = state.reshape((state.shape[0], self.train_env.physics.get_state().shape[0]))

        num_sample=600 
        idx = np.random.randint(0, state.shape[0], size=num_sample)
        state=state[idx]
        state=state.reshape(num_sample,self.train_env.physics.get_state().shape[0])
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
                obs = torch.as_tensor(obs.copy(), device=self.device).unsqueeze(0)
                z = self.agent.encoder(obs)
                encoded.append(z)
                z = self.agent.predictor(z)
                z = self.agent.projector(z)
                z = F.normalize(z, dim=1, p=2)
                proto.append(z)
                sim = self.agent.protos(z)
                idx_ = sim.argmax()
                actual_proto.append(protos[idx_][None,:])

        encoded = torch.cat(encoded,axis=0)
        proto = torch.cat(proto,axis=0)
        actual_proto = torch.cat(actual_proto,axis=0)

        sample_dist = torch.norm(proto[:,None,:] - proto[None,:, :], dim=2, p=2)
        
        if self.pmm:
            
            df = pd.DataFrame()
            df['x'] = a[:,0].round(2)
            df['y'] = a[:,1].round(2)
            df['r'] = sample_dist[0].clone().detach().cpu().numpy()
            result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['r']
            result.fillna(0, inplace=True)
            sns.heatmap(result, cmap="Blues_r",fmt='.2f', ax=ax).invert_yaxis()
            ax.set_xticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_xticklabels()])
            ax.set_yticklabels(['{:.2f}'.format(float(t.get_text())) for t in ax.get_yticklabels()])
            ax.set_title('{}, {}'.format(self.global_step, a[0,:2]))  

            plt.savefig(f"./{self.global_step}_dist_heatmap.png")
            wandb.save(f"./{self.global_step}_dist_heatmap.png")

        proto_dist = torch.norm(protos[:,None,:] - proto[None,:, :], dim=2, p=2)

        all_dists_proto, _proto = torch.topk(proto_dist, 10, dim=1, largest=False)

        #choose a random kth neighbor (k=np.random.randint(10)) of each prototype
        proto_indices = np.random.randint(10)
        
        p = _proto.clone().detach().cpu().numpy()
        
        if self.pmm:
            self.proto_goals_alt = a[p[:, proto_indices], :2]
        else:
            self.proto_goals_alt = a[p[:, proto_indices]]
            
        if self.cfg.proto_goal_intr:
            goal_dist, goal_indices = self.eval_intrinsic(encoded, a)
            dist_arg = self.proto_goals_dist.argsort(axis=0)

            for ix,x in enumerate(goal_dist.clone().detach().cpu().numpy()):
                
                if x > self.proto_goals_dist[dist_arg[ix]]:
                    
                    self.proto_goals_dist[dist_arg[ix]] = x
                    
                    if self.pmm:
                        self.proto_goals[dist_arg[ix]] = a[goal_indices[ix].clone().detach().cpu().numpy(),:2]
                    else:
                        self.proto_goals[dist_arg[ix]] = a[goal_indices[ix].clone().detach().cpu().numpy()]

            print('proto goals', self.proto_goals)
            print('proto dist', self.proto_goals_dist)
            
        elif self.cfg.proto_goal_random:
            self.proto_goals = a[_proto[:, 0].detach().clone().cpu().numpy()]


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



    def eval_pmm(self):
        #self.encode_proto(heatmap_only=True) 
        heatmaps(self, self.eval_env, self.global_step, False, True, model_step_lb=False,gc=True,proto=False)
        goal_array = self.proto_goals
        success=0
        df = pd.DataFrame(columns=['x','y','r'], dtype=np.float64) 

        for ix, x in enumerate(goal_array):
            dist_goal = cdist(np.array([x]), goal_array, 'euclidean')
            df1=pd.DataFrame()
            df1['distance'] = dist_goal.reshape((goal_array.shape[0],))
            df1['index'] = df1.index
            df1 = df1.sort_values(by='distance')
            success=0
            step, episode, total_reward = 0, 0, 0
            #goal_pix, goal_state = self.sample_goal_uniform(eval=True)
            goal_state = np.array([x[0], x[1]])
            self.eval_env = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                    self.cfg.action_repeat, seed=None, goal=goal_state)
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
                self.video_recorder.init(self.eval_env, enabled=(episode == 0))

                while step!=self.cfg.episode_length:
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        if self.cfg.goal:
                            action = self.agent.act(time_step_no_goal.observation['pixels'],
                                                time_step_goal,
                                                meta,
                                                self._global_step,
                                                eval_mode=True,
                                                tile=self.cfg.frame_stack
                                                )
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


                if ix%10==0:
                    self.video_recorder.save(f'{self.global_frame}_{ix}.mp4')

                if self.cfg.eval:
                    save(str(self.work_dir)+'/eval_{}.csv'.format(model.split('.')[-2].split('/')[-1]), [[x.cpu().detach().numpy(), total_reward, time_step.observation[:2], step]])

                else:
                        save(str(self.work_dir)+'/eval_{}.csv'.format(self._global_step), [[goal_state, total_reward, time_step.observation['observations'], step]])

                if total_reward > 20*self.cfg.num_eval_episodes:
                    success+=1

            df.loc[ix, 'x'] = x[0]
            df.loc[ix, 'y'] = x[1]
            df.loc[ix, 'r'] = total_reward
            print('r', total_reward)

        result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['r']/2
        plt.clf()
        fig, ax = plt.subplots()
        sns.heatmap(result, cmap="Blues_r").invert_yaxis()
        plt.savefig(f"./{self.global_step}_heatmap_goal.png")
        wandb.save(f"./{self.global_step}_heatmap_goal.png")

    def eval(self):

        for goal in self.proto_goals:
            print('goal', goal)
            step, episode, total_reward = 0, 0, 0
            eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
            meta = self.agent.init_meta()
            time_step_goal = self.train_env_goal.reset()
            with self.train_env_goal.physics.reset_context():

                time_step_goal = self.train_env_goal.physics.set_state(goal)

            time_step_goal = self.train_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get(self.cfg.domain, 0)) 
        
            while eval_until_episode(episode):
                time_step = self.eval_env.reset()
                self.video_recorder.init(self.eval_env, enabled=(episode == 0))
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        if self.cfg.obs_type == 'pixels' and self.pmm:
                            action = self.agent.act(time_step.observation['pixels'].copy(),
                                                    time_step_goal.copy(),
                                                    meta,
                                                    self._global_step,
                                                    eval_mode=True,
                                                    tile=self.cfg.frame_stack)
                            #non-pmm
                        elif self.cfg.obs_type == 'pixels':
                    	    action = self.agent.act(time_step.observation['pixels'].copy(),
                                                    time_step_goal.copy(),
                                                    meta,
                                                    self._global_step,
                                                    eval_mode=True,
                                                    tile=self.cfg.frame_stack) 
                        else:
                            action = self.agent.act(time_step.observation,
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

        print('r')
        if self.cfg.proto_goal_intr:
            return r, _

    
    
    def evaluate(self):
        self.logger.log('eval_total_time', self.timer.total_time(),
                        self.global_frame)
        if self.cfg.debug:
            self.eval()
        elif self.pmm:
            self.eval_proto()
            self.eval_pmm()
        else:
            self.eval_proto()
            self.eval()
        
        
    
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
            self.replay_storage.add(time_step, meta, True, pmm=self.pmm)  
        else:
            self.replay_storage.add(time_step, meta)  

        metrics = None
        
        if self.pmm:
            goal_state = self.first_goal
        else:
            goal_state = time_step.observation['observations']
            print('gs1', goal_state)
        while train_until_step(self.global_step):

            if self.global_step < self.cfg.switch_gc:
                if time_step.last():
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
                            log('buffer_size', len(self.replay_storage))
                            log('step', self.global_step)

                    if self.cfg.obs_type=='pixels':
                        time_step = self.train_env.reset()
                        print('proto', time_step.observation['observations'])
                        meta = self.agent.update_meta(meta, self._global_step, time_step)
                        self.replay_storage.add(time_step, meta, True, pmm=self.pmm)
                    else:
                        self.replay_storage.add(time_step, meta)
                    print('proto', time_step.observation['observations'])
                    # try to save snapshot
                    if self.global_frame in self.cfg.snapshots:
                        self.save_snapshot()
                    episode_step = 0
                    episode_reward = 0

                # try to evaluate
                if eval_every_step(self.global_step) and self.global_step!=0:
                    self.evaluate()

                meta = self.agent.update_meta(meta, self.global_step, time_step)
                # sample action
                with torch.no_grad(), utils.eval_mode(self.agent):
                    if self.cfg.obs_type=='pixels':
                        action = self.agent.act2(time_step.observation['pixels'],
                                        meta,
                                        self.global_step,
                                        eval_mode=True)
                    else:    
                        action = self.agent.act2(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                # try to update the agent
                if not seed_until_step(self.global_step):
                    metrics = self.agent.update(self.replay_iter, self.global_step, test=self.cfg.test)
                    self.logger.log_metrics(metrics, self.global_frame, ty='train')

#                 #save agent
                 if self._global_step%200000==0 and self._global_step!=0:
                     path = os.path.join(self.work_dir, 'optimizer_{}_{}.pth'.format(str(self.cfg.agent.name),self._global_step))
                     torch.save(self.agent, path)

                # take env step
                time_step = self.train_env.step(action)
                episode_reward += time_step.reward
                if  self.cfg.obs_type=='pixels':
                    self.replay_storage.add(time_step, meta, True, pmm=self.pmm)
                else:
                    self.replay_storage.add(time_step, meta)
                episode_step += 1
                self._global_step += 1
               
            #switching between gc & proto
            
            else:
                if self.global_step==self.cfg.switch_gc:
                    self.actor1=True
                    self.actor=False
                    print('1')
                    time_step1 = self.train_env1.reset()
                    #render first goal
                    
                    if self.pmm:
                        self.train_env_no_goal = dmc.make(self.no_goal_task, self.cfg.obs_type, self.cfg.frame_stack,
                                                            self.cfg.action_repeat, seed=None, goal=self.first_goal, 
                                                            init_state=time_step1.observation['observations'][:2])
                        time_step_no_goal = self.train_env_no_goal.reset()
                        time_step_goal = self.train_env_goal.reset()
                        with self.train_env_goal.physics.reset_context():
                        
                            time_step_goal = self.train_env_goal.physics.set_state(np.array([self.first_goal[0], self.first_goal[1], 0, 0]))

                        time_step_goal = self.train_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get(self.cfg.domain, 0))
                    #non-pmm we just use current state
                    else:
                        time_step_goal = self.train_env_goal.reset()
                        with self.train_env_goal.physics.reset_context():
                                time_step_goal = self.train_env_goal.physics.set_state(time_step.observation['observations'])
                        time_step_goal = self.train_env_goal._env.physics.render(height=84, width=84, 
                                                                                 camera_id=dict(quadruped=2).get(self.cfg.domain, 0))
                    #what does this do???
                    meta = self.agent.update_meta(meta, self._global_step, time_step1)
                    
                    if self.cfg.obs_type == 'pixels' and self.pmm:
                        self.replay_storage1.add_goal(time_step1, meta, time_step_goal, 
                                                      time_step_no_goal,self.train_env_goal.physics.state(), True)
                    elif self.cfg.obs_type == 'pixels':
                        #add function to replay buffer
                        self.replay_storage1.add_goal_general(time_step1, meta, time_step_goal, 
                                                              self.train_env_goal.physics.state(), True)
                        
                    self.eval_proto()
                
                
                if ((time_step1.last() and self.actor1) or (time_step.last() and self.actor)) and self.global_step!=self.cfg.switch_gc:
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

                    if self.cfg.obs_type =='pixels' and self.actor1 and self.pmm:
                        self.replay_storage1.add_goal(time_step1, meta,time_step_goal, time_step_no_goal, self.train_env_goal.physics.state(), True, last=True)
                    elif self.cfg.obs_type =='pixels' and self.actor1:
                        self.replay_storage1.add_goal_general(time_step1, meta, time_step_goal, self.train_env_goal.physics.state(), True, last=True)
                    elif self.cfg.obs_type =='pixels' and self.actor:
                        self.replay_storage.add(time_step, meta, True, last=True, pmm=self.pmm)
                    else:
                        self.replay_storage.add(time_step, meta)

                    # try to save snapshot
                    if self.global_frame in self.cfg.snapshots:
                        self.save_snapshot()
                    
                    episode_step = 0
                    episode_reward = 0
                    
                    #proto explores first, followed by gc, then gc resets
                    if self.proto_explore_count <= 10 and self.proto_explore:
                        
                        self.actor=True
                        self.actor1=False
                        self.proto_explore_count+=1

                    elif self.proto_explore and self.proto_explore_count > 10:
                        
                        self.actor1=True
                        self.actor=False
                        self.proto_explore=False
                        self.proto_explore_count=0
                        self.gc_explore=False
                        self.gc_explore_count=0
                        
                    elif self.gc_explore and self.gc_explore_count <= 10:

                        self.actor1=True
                        self.actor=False
                        self.proto_explore=False
                        self.proto_explore_count=0
                        self.gc_explore_count+=1
                        
                    elif self.gc_explore and self.gc_explore_count>10:
                        
                        self.actor1=True
                        self.actor=False
                        self.proto_explore=False
                        self.gc_explore=False
                        self.gc_explore_count=0
                        self.proto_explore_count=0
                        
                # try to evaluate
                if eval_every_step(self.global_step) and self.global_step!=0:
                    self.evaluate()

                if episode_step== 0 and self.global_step!=0:
                    
                    if self.proto_explore and self.actor:
                        time_step = self.train_env.reset()
                        print('proto', time_step.observation['observations'])
                        meta = self.agent.update_meta(meta, self._global_step, time_step)
                        
                        if self.cfg.obs_type == 'pixels' and time_step.last()==False:
                            self.replay_storage.add(time_step, meta, True, last=False, pmm=self.pmm)
                        
                    else:
                        print('else')
                        self.recorded=False
                        
                        ##################################################################################
                        #think about how this is going to work in high dim for reached goal frequency 
#                         if np.any(self.goal_freq==0):
#                             inv_freq = (1/(self.goal_freq+1))
#                         else:
#                             inv_freq = (1/self.goal_freq)  
#                         goal_score = np.zeros((self.proto_goals.shape[0],))

#                         for ix,x in enumerate(self.proto_goals_id):
#                             x = x.astype(int)
#                             goal_score[ix] = inv_freq[x[0], x[1]]

#                         goal_prob = F.softmax(torch.tensor(goal_score), dim=0)

#                         if self.cfg.proto_goal_intr:
#                             idx = pyd.Categorical(goal_prob).sample().item()
#                             print('goal score', goal_score)
#                             print('goal_prob', goal_prob)
#                         elif self.cfg.proto_goal_random:

                        s = self.agent.protos.weight.data.shape[0]
                        num = s+self.unreached_goals.shape[0]
                        idx = np.random.randint(num)
                        
                        if idx >= self.agent.protos.weight.data.shape[0] and self.pmm:
                    
                            goal_state = np.array([self.unreached_goals[idx-s][0], self.unreached_goals[idx-s][1]])
                        elif idx >= self.agent.protos.weight.data.shape[0]:
                            goal_state = self.unreached_goals[idx-s]
                        elif idx < self.agent.protos.weight.data.shape[0] and self.pmm:
                            goal_state = np.array([self.proto_goals[idx][0], self.proto_goals[idx][1]])
                        elif  idx < self.agent.protos.weight.data.shape[0]:
                            goal_state = self.proto_goals[idx]
                        goal_idx = idx
                        
                        if self.gc_explore:
                            print('gc exploreing')
                            print('current', self.current_init)
                            if self.pmm:
                            
                                self.train_env1 = dmc.make(self.cfg.task, self.cfg.obs_type, 
                                                       self.cfg.frame_stack,self.cfg.action_repeat, 
                                                       seed=None, goal=goal_state, init_state=self.current_init)
                            
                            else:
                                
                                time_step1 = self.train_env1.physics.set_state(self.current_init)
                                print('1')
                                print('ts', time_step1.observations['observation'])
                                print('train env', self.train_env1.physics.get_state())
                                #check when debugging to make sure stepping in this environment works 
                                print('check stepping in this env works')
                                print("code in last part of random agent pm")
                                import IPython as ipy; ipy.embed(colors='neutral')
                                #make goal 
                                
                                #####################################################
                                #follow this piece of code & make sure it's not being reset when it shouldn't be 
                            
                        else:

                            print('gc not exploring, either gc reset or proto exploring')
                            if self.pmm:
                                self.train_env1 = dmc.make(self.cfg.task, self.cfg.obs_type, 
                                                  self.cfg.frame_stack,self.cfg.action_repeat, 
                                                  seed=None, goal=goal_state)
                            else:
                                ###############################################################
                                #double check that every dmc.make() for non-pmm doesn't use goal state
                                #need to come up with a different way to calculate reward 
                                self.train_env1 = dmc.make(self.cfg.task, self.cfg.obs_type, 
                                                  self.cfg.frame_stack,self.cfg.action_repeat, 
                                                  seed=None)
                        
                        #make goals for actor 1 
                        if self.pmm:
                            time_step1 = self.train_env1.reset()

                            self.train_env_no_goal = dmc.make(self.no_goal_task, self.cfg.obs_type, self.cfg.frame_stack,
                                                            self.cfg.action_repeat, seed=None, goal=goal_state, 
                                                            init_state=time_step1.observation['observations'][:2])
                            time_step_no_goal = self.train_env_no_goal.reset()
                            time_step_goal = self.train_env_goal.reset()
                            
                            with self.train_env_goal.physics.reset_context():

                                time_step_goal = self.train_env_goal.physics.set_state(np.array([goal_state[0], goal_state[1], 0, 0]))
  
                        
                        #non-pmm
                        elif self.gc_explore:
                            #otherwise we are setting the state in this train_env1 & therefore shouldn't reset

                            time_step_goal = self.train_env_goal.reset()
                            
                            with self.train_env_goal.physics.reset_context():
                                    time_step_goal = self.train_env_goal.physics.set_state(
                                                        np.array(goal_state))
                                    ##change eval_proto so that it saves pos + vel as proto_goals
                                    
                        elif self.gc_explore==False and self.pmm==False:
                            
                            time_step1 = self.train_env1.reset()

                            time_step_goal = self.train_env_goal.reset()
                            
                            with self.train_env_goal.physics.reset_context():

                                time_step_goal = self.train_env_goal.physics.set_state(goal_state)
                        
                        
                        #this part is same for both pmm & non-pmm
                        time_step_goal = self.train_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get(self.cfg.domain, 0))   
                        meta = self.agent.update_meta(meta, self._global_step, time_step1) 
                        print('time step', time_step1.observation['observations'])
                        print('sampled goal', goal_state)
                        

                        if self.cfg.obs_type == 'pixels' and time_step1.last()==False and episode_step!=self.cfg.episode_length and self.pmm:
                            self.replay_storage1.add_goal(time_step1, meta, time_step_goal, 
                                                          time_step_no_goal,self.train_env_goal.physics.state(), True, pmm=self.pmm)
                            
                        elif self.cfg.obs_type == 'pixels' and time_step1.last()==False and episode_step!=self.cfg.episode_length:
                            
                            self.replay_storage1.add_goal_general(time_step1, meta,time_step_goal,
                                                                  self.train_env_goal.physics.state(), True)
                            
                
                # sample action
                if self.actor1:
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        if self.cfg.obs_type == 'pixels' and self.pmm:
                            action1 = self.agent.act(time_step_no_goal.observation['pixels'].copy(),
                                                    time_step_goal.copy(),
                                                    meta,
                                                    self._global_step,
                                                    eval_mode=False,
                                                    tile=self.cfg.frame_stack)
                        #non-pmm
                        elif self.cfg.obs_type == 'pixels':
                            action1 = self.agent.act(time_step1.observation['pixels'].copy(),
                                                    time_step_goal.copy(),
                                                    meta,
                                                    self._global_step,
                                                    eval_mode=False,
                                                    tile=self.cfg.frame_stack)
                        else:
                            action = self.agent.act(time_step.observation,
                                                meta,
                                                self.global_step,
                                                eval_mode=False,
                                                tile=self.cfg.frame_stack)

                    # take env step
                    time_step1 = self.train_env1.step(action1)
                    if self.pmm:
                        time_step_no_goal = self.train_env_no_goal.step(action1)
                    episode_reward += time_step1.reward

                    if self.cfg.obs_type == 'pixels' and time_step1.last()==False and episode_step!=self.cfg.episode_length and self.cfg.resample_goal==False and self.pmm:
                        
                        self.replay_storage1.add_goal(time_step1, meta, time_step_goal, 
                                                      time_step_no_goal,self.train_env_goal.physics.state(), True)
                        
                    elif self.cfg.obs_type == 'pixels' and time_step1.last()==False and episode_step!=self.cfg.episode_length and self.cfg.resample_goal==False:
                        
                        self.replay_storage1.add_goal_general(time_step1, meta, time_step_goal, 
                                                      self.train_env_goal.physics.state(), True)
                        
                    elif self.cfg.obs_type == 'pixels' and time_step1.last()==False and episode_step!=self.cfg.episode_length and self.cfg.resample_goal and episode_reward <= 100 and self.pmm:
                        
                        self.replay_storage1.add_goal(time_step1, meta, time_step_goal, 
                                                      time_step_no_goal, self.train_env_goal.physics.state(), True)
                        
                    elif self.cfg.obs_type == 'pixels' and time_step1.last()==False and episode_step!=self.cfg.episode_length and self.cfg.resample_goal and episode_reward <= 100:
                        
                        self.replay_storage1.add_goal_general(time_step1, meta, time_step_goal, 
                                                              self.train_env_goal.physics.state(), True)
                        
                    elif self.cfg.obs_type == 'states':
                        self.replay_storage1.add_goal(time_step1, meta, goal)
                
                #self.actor. proto's turn
                else:
                    
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        
                        if self.cfg.obs_type=='pixels':

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
                     
                    time_step = self.train_env.step(action)
                    episode_reward += time_step.reward
                    
                    if  self.cfg.obs_type=='pixels' and time_step.last()==False:
                        self.replay_storage.add(time_step, meta, True, pmm=self.pmm)

                    elif time_step.last()==False and self.cfg.obs_type=='states':
                        self.replay_storage.add(time_step, meta)

                episode_step += 1

                #sample new goals if episode reward > 100 
                if self.actor1:
                    
                    #change to vicinity of reached_goals
                    if (goal_state in self.reached_goals and (time_step1.reward > 1.7) and (self.proto_explore==False) and (episode_step> 20)) or (episode_reward > 100):
                        
                        print('proto_goals', self.proto_goals)
                        if self.pmm:
                            idx_x = int(goal_state[0]*100)+29
                            idx_y = int(goal_state[1]*100)+29
                            self.proto_goals_matrix[idx_x,idx_y]+=1
                            self.goal_freq[idx_x//10, idx_y//10]+=1

                        self.reached_goals = np.append(self.reached_goals, goal_state[None,:], axis=0)
                        self.reached_goals = np.unique(self.reached_goals, axis=0)
                        
                        ##first reset gc agent from here and try to reach the nearby goals for 10 episodes 
                        if self.pmm:
                            min_dist = np.argmin(np.linalg.norm(np.tile(time_step1.observation['observations'][:2], (self.proto_goals_alt.shape[0],1)) - self.proto_goals_alt))
                            
                            if self.reached_goals.shape[0]!=0:
                                goal_dist = min(np.linalg.norm(self.proto_goals_alt[min_dist][None,:] - self.reached_goals, ord=2, axis=1))
                                if goal_dist < .02:
                                    self.proto_goals_alt = np.delete(self.proto_goals_alt, 0, min_dist)
                                    min_dist = np.argmin(np.linalg.norm(np.tile(time_step1.observation['observations'][:2], (self.proto_goals_alt.shape[0],1)) - self.proto_goals_alt))
                                
                        else:
                            print("min_dist = np.argmin(np.linalg.norm(np.tile(time_step1.observation['observations'], (self.proto_goals_alt.shape[0],1)) - self.proto_goals_alt))")
                            import IPython as ipy; ipy.embed(colors='neutral')
                            min_dist = np.argmin(np.linalg.norm(np.tile(time_step1.observation['observations'], (self.proto_goals_alt.shape[0],1)) - self.proto_goals_alt))

                            if self.reached_goals.shape[0]!=0:
                                goal_dist = min(np.linalg.norm(self.proto_goals_alt[min_dist][None,:] - self.reached_goals, ord=2, axis=1))
                                #may need to change this threshold in high dim
                                if goal_dist < .02:
                                    print('goaldist < .02, investigate')
                                    print("min(np.linalg.norm(self.proto_goals_alt[min_dist][None,:] - self.reached_goals, ord=2, axis=1))")
                                    import IPython as ipy; ipy.embed(colors='neutral')
                                    self.proto_goals_alt = np.delete(self.proto_goals_alt, 0, min_dist)
                                    min_dist = np.argmin(np.linalg.norm(np.tile(time_step1.observation['observations'], (self.proto_goals_alt.shape[0],1)) - self.proto_goals_alt))

                        goal_state = self.proto_goals_alt[min_dist]
                        print('gs2', goal_state) 
                        if goal_state in self.proto_goals:
                            goal_idx=np.where((self.proto_goals == goal_state).all(axis=1))[0]
                        else:
                            goal_idx=np.inf
                        print('new goal', goal_state) 
                        print('sampling new goal')
                        episode_reward=0
                        episode_step=0
                        
                        self.proto_explore=True
                        print('proto explore')

                        if self.cfg.obs_type == 'pixels' and time_step1.last()==False and self.pmm:
                            self.replay_storage1.add_goal(time_step1, meta,time_step_goal, 
                                                          time_step_no_goal, self.train_env_goal.physics.state(), True, last=True)
                            self.current_init = time_step1.observation['observations'][:2]
                            
                            meta = self.agent.update_meta(meta, self._global_step, time_step1)

                            self.train_env = dmc.make(self.cfg.task_no_goal, self.cfg.obs_type, self.cfg.frame_stack,
                                                  self.cfg.action_repeat, seed=None, goal=goal_state,
                                                  init_state=self.current_init)
                        
                        elif self.cfg.obs_type == 'pixels' and time_step1.last()==False:
                            self.replay_storage1.add_goal_general(time_step1, meta,time_step_goal,
                                                                  self.train_env_goal.physics.state(), True, last=True)
                            self.current_init = time_step1.observation['observations']
                            
                            meta = self.agent.update_meta(meta, self._global_step, time_step1)
                            
                            with self.train_env1.physics.reset_context():
                        
                                time_step1 = self.train_env1.physics.set_state(self.current_init)
                            

                        time_step = self.train_env.reset()
                        meta = self.agent.update_meta(meta, self._global_step, time_step)
                        print('should start proto')

                    if self.global_step > (self.cfg.switch_gc+self.cfg.num_seed_frames):

                        metrics = self.agent.update(self.replay_iter1, self.global_step, actor1=True)
                        self.logger.log_metrics(metrics, self.global_frame, ty='train')
                        metrics = self.agent.update(self.replay_iter, self.global_step, test=self.cfg.test)
                        #self.logger.log_metrics(metrics, self.global_frame, ty='train')

                self._global_step += 1
            
            if self._global_step%100000==0 and self._global_step!=0:
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

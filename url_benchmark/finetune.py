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
torch.backends.cudnn.benchmark = True
from dmc_benchmark import PRIMAL_TASKS

def make_agent(obs_type, obs_spec, action_spec, goal_shape, num_expl_steps, cfg, lr=.0001, hidden_dim=1024, num_protos=512, update_gc=2, gc_only=False, offline=False, tau=.1, num_iterations=3, feature_dim=50, pred_dim=128, proj_dim=512, batch_size=1024, update_proto_every=10, lagr=.2, margin=.5, lagr1=.2, lagr2=.2, lagr3=.3, stddev_schedule=.2, stddev_clip=.3, update_proto=2, stddev_schedule2=.2, stddev_clip2=.3, update_enc_proto=False, update_enc_gc=False):

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
            goals = np.array([[.15, .15], [-.15, .15], [-.15, -.15], [.15, -.15], [.2, .2], [.2, -.05]])
            self.finetune_goal = goals[self.cfg.goal_index]
            self.train_env1 = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                       cfg.action_repeat, seed=None, goal=self.finetune_goal)
            print('goal', self.finetune_goal)
            self.train_env_no_goal = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                       cfg.action_repeat, seed=None, goal=None)
            print('no goal task env', self.no_goal_task)
            self.train_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                      cfg.action_repeat, seed=None, goal=self.finetune_goal)
            self.eval_env = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                     cfg.action_repeat, seed=None, goal=self.finetune_goal)
            self.eval_env_no_goal = dmc.make(self.no_goal_task, cfg.obs_type, cfg.frame_stack,
                                     cfg.action_repeat, seed=None, goal=None)
        else:
            self.train_env1 = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                       cfg.action_repeat, seed=None)
            self.train_env = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                                      cfg.action_repeat, seed=None)
            self.eval_env = dmc.make(self.cfg.task, cfg.obs_type, cfg.frame_stack,
                                     cfg.action_repeat, seed=None)

        if cfg.cassio:
            self.pwd = '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark'
        elif cfg.greene:
            self.pwd = '/vast/nm1874/dm_control_2022/proto_explore/url_benchmark'
        
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
                                stddev_clip2=cfg.stddev_clip2,
                                update_enc_proto=cfg.update_enc_proto,
                                update_enc_gc=cfg.update_enc_gc)     
        # initialize from pretrained

        if cfg.model_path is not None:
            pretrained_agent = torch.load(self.pwd + cfg.model_path)
            self.agent.init_from(pretrained_agent)

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
        self.replay_loader1 = make_replay_loader(self.replay_storage1,
                                                    False,
                                                    cfg.replay_buffer_gc,
                                                    cfg.batch_size_gc,
                                                    cfg.replay_buffer_num_workers,
                                                    False, cfg.nstep1, cfg.discount,
                                                    True, cfg.hybrid_gc,cfg.obs_type,
                                                    cfg.hybrid_pct, loss=cfg.loss_gc, test=cfg.test,
                                                    tile=cfg.frame_stack,
                                                    pmm=self.pmm,
                                                    obs_shape=self.train_env1.physics.state().shape[0],
                                                    general=True)
        
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                False,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep2, cfg.discount,
                                                goal=False,
                                                obs_type=cfg.obs_type,
                                                loss=cfg.loss,
                                                test=cfg.test) 
        self._replay_iter = None
        self._replay_iter1 = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None)

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
        if self.cfg.proto_goal_intr:
            dim=10
        else:
            dim=self.agent.protos.weight.data.shape[0]

        self.proto_goals = np.zeros((dim, 3*self.cfg.frame_stack, 84, 84))
        self.proto_goals_state = np.zeros((dim, self.train_env.physics.get_state().shape[0]))
        self.proto_goals_matrix = np.zeros((60,60))
        self.proto_goals_id = np.zeros((dim, 2))
        self.actor=True
        self.actor1=False
        self.final_df = pd.DataFrame(columns=['avg', 'med', 'max', 'q7', 'q8', 'q9'])
        self.reached_goals=np.empty((0,self.train_env.physics.get_state().shape[0]))
        self.proto_goals_alt=[]
        self.proto_explore=False
        self.proto_init =False
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
        self.unreached_goals = np.empty((0,self.train_env.physics.get_state().shape[0]))
        self.proto_last_explore=0	   
        self.current_init = []
        self.unreached = False
        self.df = pd.DataFrame(columns=['x','y','r'], dtype=np.float64) 

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
    @property
    def replay_iter1(self):
        if self._replay_iter1 is None:
            self._replay_iter1 = iter(self.replay_loader1)
        return self._replay_iter1

  
    def eval_proto(self):
        
        if self.pmm and self.global_step%100000==0:
            heatmaps(self, self.eval_env, self.global_step, False, True, model_step_lb=False,gc=True,proto=True)
            eval_env_goal = dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None)
        
        ########################################################################
        #how should we measure exploration in non-pmm w/o heatmaps

        
        protos = self.agent.protos.weight.data.detach().clone()

        
        #####################################################################
        #probably need to cut down .5mil in buffer for non-pmm
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
                                                model_step = 1000000,
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

        proto_dist = torch.norm(protos[:,None,:] - proto[None,:, :], dim=2, p=2)

        all_dists_proto, _proto = torch.topk(proto_dist, 10, dim=1, largest=False)

        #choose a random kth neighbor (k=np.random.randint(10)) of each prototype
        proto_indices = np.random.randint(10)
        
        p = _proto.clone().detach().cpu().numpy()
            
        
            
        if self.cfg.proto_goal_random:
            
            closest_sample = _proto[:, 0].detach().clone().cpu().numpy()
            print('closest', closest_sample.shape) 
            for ix, x in enumerate(closest_sample):
                
                fn = eps[x]
                idx_ = index[x]
                ep = np.load(fn)
                #pixels.append(ep['observation'][idx_])

                with torch.no_grad():
                    obs = ep['observation'][idx_]
                    
                self.proto_goals[ix] = obs
            self.proto_goals_state = a[closest_sample]

            print('goal pix1', self.proto_goals.shape)
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
            else:
                print('add tsne plot for higher dim env')
                
        ########################################################################
        #implement tsne for non-pmm prototype eval?
        if self.global_step%100000==0: 
            for ix in range(self.proto_goals_state.shape[0]):
                 
                with self.eval_env.physics.reset_context():
                    self.eval_env.physics.set_state(self.proto_goals_state[ix])
                
                plt.clf()
                img = self.eval_env._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get(self.cfg.domain, 0))
                plt.imsave(f"goals_{ix}_{self.global_step}.png", img)
                wandb.save(f"goals_{ix}_{self.global_step}.png")
                
        index=np.where(((np.linalg.norm(self.proto_goals[:,None,:] - self.current_init[None,:,:],axis=-1, ord=2))<.05))
        index = np.unique(index[0])
        print('delete goals', self.proto_goals[index])
        self.proto_goals = np.delete(self.proto_goals, index,axis=0)
        self.proto_goals_dist = np.delete(self.proto_goals_dist, index,axis=0)
        index=np.where((self.proto_goals==0.).all(axis=1))[0]
        self.proto_goals = np.delete(self.proto_goals, index,axis=0)
        self.proto_goals_dist = np.delete(self.proto_goals_dist, index,axis=0)
        print('current goals', self.proto_goals)   
        
        
    
    def eval(self):
        
        if self.pmm and self.global_step%100000==0:
            heatmaps(self, self.eval_env, self.global_step, False, True, model_step_lb=False,gc=True,proto=True)
            

        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):

                        action = self.agent.act2(time_step.observation['pixels'],
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
        time_step1 = self.train_env1.reset()
        self.actor1=True
        self.actor=False
        if self.pmm:
            self.train_env_no_goal = dmc.make(self.no_goal_task, self.cfg.obs_type, self.cfg.frame_stack,
                                                        self.cfg.action_repeat, seed=None, goal=None, 
                                                        init_state=time_step1.observation['observations'][:2])
                    
            time_step_no_goal = self.train_env_no_goal.reset()

        meta = self.agent.init_meta()
        self.train_video_recorder.init(time_step.observation)
        
        if self.cfg.obs_type == 'pixels':
            self.replay_storage.add(time_step, self.train_env.physics.get_state(), meta, True, pmm=self.pmm)  
        else:
            self.replay_storage.add(time_step, meta=meta)  
            
        metrics = None
        
        goal_idx = 0 
        
        if self.pmm==False:
            time_step_no_goal = None
            
        self.eval_proto()
        goal_idx = 0
        goal_state = self.proto_goals_state[goal_idx]
        goal_pix = self.proto_goals[goal_idx]

        while train_until_step(self.global_step):
            if ((time_step1.last() and self.actor1) or (time_step.last() and self.actor)):
                print('last')
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4')
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
                        
                if self.cfg.obs_type =='pixels' and self.actor1:
                    self.replay_storage1.add_goal_general(time_step1, self.train_env1.physics.get_state(), meta, 
                                                          goal_pix, goal_state, 
                                                          time_step_no_goal, True, last=True)


                elif self.cfg.obs_type =='pixels' and self.actor:
                    self.replay_storage.add(time_step, self.train_env.physics.get_state(), meta, True, last=True, pmm=self.pmm)
                else:
                    self.replay_storage.add(time_step, meta=meta)
                    
                if self.proto_explore==False:
                    self.proto_last_explore+=1
                else:
                    self.proto_last_explore=0
                
                if self.proto_explore_count > 5:
                    self.proto_init=True
                      
                        
                self.unreached=False

                episode_step = 0
                episode_reward = 0
                
                if self.proto_explore_count <= 20 and self.proto_explore:
                        
                    self.actor=True
                    self.actor1=False
                    self.proto_explore_count+=1

                elif self.proto_explore and self.proto_explore_count > 20:

                    self.actor1=True
                    self.actor=False
                    self.proto_explore=False
                    self.proto_explore_count=0
                    self.gc_explore=True
                    self.gc_explore_count=0

                elif self.gc_explore and self.gc_explore_count <= 20:

                    self.actor1=True
                    self.actor=False
                    self.proto_explore=False
                    self.proto_explore_count=0
                    self.gc_explore_count+=1

                elif self.gc_explore and self.gc_explore_count>20:

                    self.actor1=True
                    self.actor=False
                    self.proto_explore=False
                    self.gc_explore=False
                    self.gc_explore_count=0
                    self.proto_explore_count=0

            # try to evaluate
            if eval_every_step(self.global_step) and self.global_step%100000:
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
               
                self.eval()
                

            meta = self.agent.update_meta(meta, self.global_step, time_step)

            ###################################
            #what is this for??
            if hasattr(self.agent, "regress_meta"):
                repeat = self.cfg.action_repeat
                every = self.agent.update_task_every_step // repeat
                init_step = self.agent.num_init_steps
                if self.global_step > (
                        init_step // repeat) and self.global_step % every == 0:
                    meta = self.agent.regress_meta(self.replay_iter,
                                                   self.global_step)
                    
            if episode_step== 0:

                if self.proto_last_explore > 100 and self.gc_explore==False:

                    self.proto_explore=True
                    self.actor=True
                    self.actor1=False
                    self.proto_last_explore=0
                    print('proto last >100')

                if self.proto_explore and self.actor:

                    #now the proto explores from any reached goals by gc
                    if len(self.current_init) > 0:

                        init_idx=np.random.randint(len(self.current_init))

                        time_step = self.train_env.reset()
                        ##can't reset now
                        with self.train_env.physics.reset_context():
                            self.train_env.physics.set_state(self.current_init[init_idx])
                        act_ = np.zeros(self.train_env.action_spec().shape, self.train_env.action_spec().dtype)
                        time_step = self.train_env.step(act_)
                    else:

                        print('no current init yet')
                        if self.pmm:
                            rand_init = np.random.uniform(.25,.29,size=(2,))
                            rand_init[0] = rand_init[0]*(-1)

                            self.train_env = dmc.make(self.cfg.task, self.cfg.obs_type,
                                                       self.cfg.frame_stack,self.cfg.action_repeat,
                                                       seed=None, goal=goal_state[:2], init_state = rand_init) 

                        time_step = self.train_env.reset()


                    print('proto_explore', time_step.observation['observations'])
                   

                    if self.cfg.obs_type == 'pixels' and time_step.last()==False:
                        self.replay_storage.add(time_step, self.train_env.physics.get_state(), meta, True, last=False, pmm=self.pmm)

                else:
                    print('else')
                    self.recorded=False

                    ###################################
                    #sampling goal here
                    s = self.proto_goals.shape[0]
                    num = s+self.unreached_goals.shape[0]
                    idx = np.random.randint(num)

                    if idx >= s:

                        goal_idx = idx-s
                        goal_state = self.unreached_goals[goal_idx]
                        with self.eval_env_no_goal.physics.reset_context():
                            self.eval_env_no_goal.physics.set_state(goal_state)
                        goal_pix = self.eval_env_no_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
                        goal_pix = np.transpose(goal_pix, (2,0,1))
                        goal_pix = np.tile(goal_pix, (self.cfg.frame_stack,1,1))
                        self.unreached=True


                    else:
                        goal_idx = idx
                        goal_state = self.proto_goals_state[goal_idx]
                        goal_pix = self.proto_goals[goal_idx]

                    print('goal pix', goal_pix.shape)
                    #v1: gc always starts from the most recently reached goal 

                    #if self.cfg.test1:
                    print('gc ALWYAS exploreing')
                    if len(self.current_init) != 0:

                        init_idx=np.random.randint(len(self.current_init))
                        time_step1 = self.train_env1.reset()
                        with self.train_env1.physics.reset_context():
                            self.train_env1.physics.set_state(self.current_init[init_idx])
                        act_ = np.zeros(self.train_env.action_spec().shape, self.train_env.action_spec().dtype)
                        time_step1 = self.train_env1.step(act_)
                    else:

                        print('no current init yet')
                        if self.pmm:

                            rand_init = np.random.uniform(.25,.29,size=(2,))
                            rand_init[0] = rand_init[0]*(-1)
                            print('r', rand_init)
                            print('g', goal_state)

                            self.train_env1 = dmc.make(self.cfg.task, self.cfg.obs_type,
                                               self.cfg.frame_stack,self.cfg.action_repeat,
                                               seed=None, goal=goal_state[:2], init_state = rand_init) 

                        time_step1 = self.train_env1.reset()


                    if self.pmm:
                        self.train_env_no_goal = dmc.make(self.no_goal_task, self.cfg.obs_type, self.cfg.frame_stack,
                                self.cfg.action_repeat, seed=None, goal=goal_state[:2], 
                                                        init_state=time_step1.observation['observations'][:2])

                        time_step_no_goal = self.train_env_no_goal.reset()
                    
                    print('time step', time_step1.observation['observations'])
                    print('sampled goal', goal_idx)

                   #unreached_goals : array of states (not pixel), need to set state & render to get pixel
                    if self.cfg.obs_type == 'pixels' and time_step1.last()==False and episode_step!=self.cfg.episode_length:
                        self.replay_storage1.add_goal_general(time_step1, self.train_env1.physics.get_state(), meta,
                                                              goal_pix, goal_state,
                                                              time_step_no_goal, True)

        # sample action
            if self.actor1:
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
                    else:
                        action1 = self.agent.act(time_step.observation,
                                            meta,
                                            self.global_step,
                                            eval_mode=False,
                                            tile=self.cfg.frame_stack)

                        
                # take env step

                time_step1 = self.train_env1.step(action1)
                if self.pmm:
                    time_step_no_goal = self.train_env_no_goal.step(action1)
                    
                #calculate reward
                if self.cfg.ot_reward:

                    reward = self.agent.ot_rewarder(
                         time_step1.observation['pixels'], goal_pix, self.global_step)

#                     elif self.cfg.dac_reward:

#                         reward = self.agent.dac_rewarder(time_step1.observation['pixels'], action1)

                elif self.cfg.neg_euclid:

                    with torch.no_grad():
                        obs = time_step1.observation['pixels']
                        obs = torch.as_tensor(obs.copy(), device=self.device).unsqueeze(0)
                        z1 = self.agent.encoder(obs)
                        z1 = self.agent.predictor(z1)
                        z1 = self.agent.projector(z1)

                        z1 = F.normalize(z1, dim=1, p=2)
                        if self.unreached==False: 
                            goal = torch.as_tensor(goal_pix, device=self.device).unsqueeze(0).int()
                        else:
                            goal = torch.as_tensor(tmp_unreached.copy(), device=self.device).unsqueeze(0).int()
                        z2 = self.agent.encoder(goal)
                        z2 = self.agent.predictor(z2)
                        z2 = self.agent.projector(z2)
                        z2 = F.normalize(z2, dim=1, p=2)
                    reward = -torch.norm(z1-z2, dim=-1, p=2).item()
                elif self.cfg.neg_euclid_state:

                    reward = -np.linalg.norm(self.train_env1.physics.get_state() - goal_state, axis=-1, ord=2)
                elif self.cfg.actionable:

                    print('not implemented yet: actionable_reward')
                
                
                if self.pmm==False:
                    time_step1 = time_step1._replace(reward=reward)
                    
                if time_step1.reward > self.cfg.reward_cutoff:
                    episode_reward += abs(time_step1.reward)
                    
                if self.cfg.obs_type == 'pixels' and time_step1.last()==False and episode_step!=self.cfg.episode_length and ((episode_reward < abs(self.cfg.reward_cutoff*20) and self.pmm==False) or (episode_reward < 100 and self.pmm)):
                        
                    self.replay_storage1.add_goal_general(time_step1, self.train_env1.physics.get_state(), meta, 
                                                          goal_pix, goal_state, 
                                                  time_step_no_goal, True)



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
                    self.replay_storage.add(time_step, self.train_env.physics.get_state(), meta, True, pmm=self.pmm)

                elif time_step.last()==False and self.cfg.obs_type=='states':
                    self.replay_storage.add(time_step, meta)


            if self.actor1:

                #change to vicinity of reached_goals

                if time_step1.reward == 0.:
                    time_step1 = time_step1._replace(reward=-1)

                if ((episode_reward > abs(self.cfg.reward_cutoff*20) and self.pmm==False) or (episode_reward > 100 and self.pmm)) and episode_step>5:
                    self.unreached=False
                    print('r', episode_reward)

                    #reached_goals = array of indices corresponding to prototypes
                    self.reached_goals = np.append(self.reached_goals, self.train_env1.physics.get_state()[None,:], axis=0)
                    self.reached_goals = np.unique(self.reached_goals, axis=0)
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
                    
                    episode_reward=0
                    episode_step=0

                    self.proto_explore=True
                    self.gc_explore = False
                    self.actor = True
                    self.actor1 = False
                    print('proto explore')

                    if self.cfg.obs_type == 'pixels' and time_step1.last()==False:
                        print('last1')
                        self.replay_storage1.add_goal_general(time_step1, self.train_env1.physics.get_state(), meta, 
                                                              goal_pix, goal_state,
                                                      time_step_no_goal, True, last=True)
                        

                   


                    meta = self.agent.update_meta(meta, self._global_step, time_step1)

                    self.current_init.append(self.train_env1.physics.get_state())
                    print('current', self.current_init)
                    print('obs', self.train_env1.physics.get_state())
                    meta = self.agent.update_meta(meta, self._global_step, time_step1)


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

                if episode_step==499 and ((episode_reward < abs(self.cfg.reward_cutoff*20) and self.pmm==False) or (episode_reward < 100 and self.pmm)):
                    #keeping this for now so the unreached list doesn't grow too fast
                    if self.unreached==False:
                        self.unreached_goals=np.append(self.unreached_goals, self.proto_goals_state[goal_idx][None,:], axis=0)

            if self.global_step > (self.cfg.num_seed_frames):

                metrics = self.agent.update(self.replay_iter1, self.global_step, actor1=True)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
                if self.proto_init:
                    metrics = self.agent.update(self.replay_iter, self.global_step, test=self.cfg.test)
                #self.logger.log_metrics(metrics, self.global_frame, ty='train')
            episode_step += 1
            self._global_step += 1

        if self._global_step%10000==0 and self._global_step!=0:
                print('saving agent')
                path = os.path.join(self.work_dir, 'optimizer_{}_{}.pth'.format(str(self.cfg.agent.name),self._global_step))
                torch.save(self.agent, path)  

    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        domain, _ = self.cfg.task.split('_', 1)
        snapshot_dir = snapshot_base_dir / self.cfg.obs_type / domain / self.cfg.agent.name

        def try_load(seed):
            snapshot = snapshot_dir / str(
                seed) / f'snapshot_{self.cfg.snapshot_ts}.pt'
            if not snapshot.exists():
                return None
            with snapshot.open('rb') as f:
                payload = torch.load(f)
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        # otherwise try random seed
        while True:
            seed = np.random.randint(1, 11)
            payload = try_load(seed)
            if payload is not None:
                return payload
        return None


@hydra.main(config_path='.', config_name='finetune')
def main(cfg):
    from finetune import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()

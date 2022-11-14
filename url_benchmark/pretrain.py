import scipy
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
import pandas as pd
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
import dmc
import utils
from scipy.spatial.distance import cdist
from logger import Logger, save
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_replay_buffer, ndim_grid, make_replay_offline
import matplotlib.pyplot as plt
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS
import random
import imageio
from textwrap import wrap
import itertools
import seaborn as sns; sns.set_theme()
from pathlib import Path
import io

def make_agent(obs_type, obs_spec, action_spec, goal_shape, num_expl_steps, cfg, lr=.0001, hidden_dim=1024, num_protos=512, update_gc=2, gc_only=False, offline=False, tau=.1, num_iterations=3, feature_dim=50, pred_dim=128, proj_dim=512, batch_size=1024, update_proto_every=10, lagr=.2, margin=.5, lagr1=.2, lagr2=.2, lagr3=.3):

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
    if cfg.name=='protox':
        cfg.lagr1 = lagr1
        cfg.lagr2 = lagr2
        cfg.lagr3 = lagr3

    if cfg.name=='protov2':
        cfg.update_proto_every=update_proto_every
    return hydra.utils.instantiate(cfg)

def heatmaps(self, model_step):



    heatmap = self.replay_storage.state_visitation_proto

    plt.clf()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
    ax.set_title(model_step)

    plt.savefig(f"./{model_step}_proto_heatmap.png")
    wandb.save(f"./{model_step}_proto_heatmap.png")


    heatmap_pct = self.replay_storage.state_visitation_proto_pct

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
        self.train_env = dmc.make(self.cfg.task_no_goal, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_no_goal, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, cfg.seed)

        # create agent
        if self.cfg.agent.name=='protov2':
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
                                update_proto_every=cfg.update_proto_every)
        
        elif self.cfg.agent.name=='protox':
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
                                margin=cfg.margin) 
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
                                margin=cfg.margin)

        
        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer')

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                False,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount,
                                                goal=cfg.goal,
                                                obs_type=cfg.obs_type,
                                                loss=cfg.loss,
                                                test=cfg.test)
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
        self.final_df = pd.DataFrame(columns=['avg', 'med', 'max', 'q7', 'q8', 'q9'])

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
    
    
    def eval_protov2(self):
        heatmaps(self, self.global_step)
        eval_env_goal = dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None)

        protos = self.agent.protos.detach().clone().cpu().numpy()
        replay_buffer = make_replay_offline(eval_env_goal,
                                                self.work_dir / 'buffer' / 'buffer_copy',
                                                100000,
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
        state = state.reshape((state.shape[0],4))
        print(state.shape)
        
        if self.global_step<=10000:
            num_sample=self.global_step//2
        else:
            num_sample=10000
        state_t = np.empty((num_sample,4))
        proto_t = np.empty((num_sample,protos.shape[1]))

        encoded = []
        proto = []
        actual_proto = []
        lst_proto = []
        
        idx = np.random.choice(state.shape[0], size=num_sample, replace=False)
        print('starting to load 50k')
        for ix,x in enumerate(idx):
            print(ix)
            state_t[ix] = state[x]
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
                proto_t[ix]=z.cpu().numpy()


        print('data loaded in',state.shape[0])

        covar = np.cov(proto_t.T)
        print(covar.shape)
        U, S, Vh = scipy.linalg.svd(covar)
        print(S)
        plt.plot(S)
        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(S)
        ax.set_title('singular values')
        plt.savefig(self.work_dir / f"singular_value_{self.global_step}.png")


        num_sample=1000 
        idx = np.random.randint(0, state.shape[0], size=num_sample)
        state=state[idx]
        state=state.reshape(num_sample,4)
        a = state
        count10,count01,count00,count11=(0,0,0,0)
        # density estimate:
        df = pd.DataFrame()
        for state_ in a:
            if state_[0]<0:
                if state_[1]>=0:
                    count10+=1
                else:
                    count00+=1
            else:
                if state_[1]>=0:
                    count11+=1
                else:
                    count01+=1

        df.loc[0,0] = count00/a.shape[0]
        df.loc[0,1] = count01/a.shape[0]
        df.loc[1,1] = count11/a.shape[0]
        df.loc[1,0] = count10/a.shape[0]
        labels=df
        plt.clf()
        fig, ax = plt.subplots()
        sns.heatmap(df, cmap="Blues_r",cbar=False, annot=labels).invert_yaxis()
        ax.set_title('data percentage')
        plt.savefig(self.work_dir / f"data_pct_model_{self._global_step}.png")

        def ndim_grid(ndims, space):
            L = [np.linspace(-.25,.25,space) for i in range(ndims)]
            return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T

        lst=[]
        goal_array = ndim_grid(2,10)
        for ix,x in enumerate(goal_array):
            if (-.2<x[0]<.2 and -.02<x[1]<.02) or (-.02<x[0]<.02 and -.2<x[1]<.2):
                lst.append(ix)


        goal_array=np.delete(goal_array, lst,0)
        emp = np.zeros((goal_array.shape[0],2))
        goal_array = np.concatenate((goal_array, emp), axis=1)


        ##########################################################################################################################        

#         ##encoded goals w/ no velocity 

#         actual_proto_no_v=[]
#         encoded_no_v=[]
#         proto_no_v = []
#         #no velocity goals 
#         actual_proto_no_v = []
#         goal_array = ndim_grid(2,10)
#         for ix,x in enumerate(goal_array):
#             if (-.2<x[0]<.2 and -.02<x[1]<.02) or (-.02<x[0]<.02 and -.2<x[1]<.2):
#                 lst.append(ix)
#         goal_array=np.delete(goal_array, lst,0)

#         lst_proto = []
#         for x in goal_array:
#             with torch.no_grad():
#                 with eval_env_goal.physics.reset_context():
#                     time_step_init = eval_env_goal.physics.set_state(np.array([x[0].item(), x[1].item(),0,0]))

#                 time_step_init = eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
#                 time_step_init = np.transpose(time_step_init, (2,0,1))
#                 time_step_init = np.tile(time_step_init, (3,1,1))

#                 obs = time_step_init
#                 obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
#                 z = self.agent.encoder(obs)
#                 encoded_no_v.append(z)
#                 z = self.agent.predictor(z)
#                 z = self.agent.projector(z)
#                 z = F.normalize(z, dim=1, p=2)
#                 proto_no_v.append(z)
#                 sim = torch.mm(self.agent.protos.detach().clone(), z.T)
#                 print('sim',sim.shape)
#                 idx_ = sim.argmax()
#                 lst_proto.append(idx_)
#                 actual_proto_no_v.append(self.agent.protos.detach().clone()[idx_][None,:])

#         print('ndim_grid no velocity: therere {} unique prototypes that are neighbors to {} datapoints'.format(len(set(lst_proto)), goal_array.shape[0]))

#         encoded_no_v = torch.cat(encoded_no_v,axis=0)
#         proto_no_v = torch.cat(proto_no_v,axis=0)
#         actual_proto_no_v = torch.cat(actual_proto_no_v,axis=0)

        #pixels = []
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
                sim = torch.mm(self.agent.protos.detach().clone(), z.T)
                idx_ = sim.argmax()
                actual_proto.append(self.agent.protos.detach().clone()[idx_][None,:])

        encoded = torch.cat(encoded,axis=0)
        proto = torch.cat(proto,axis=0)
        actual_proto = torch.cat(actual_proto,axis=0)

################################################################


#         #no velocity goals 


#         encoded_no_vdist = torch.norm(encoded_no_v[:,None, :] - encoded[None,:, :], dim=2, p=2)
#         proto_no_vdist = torch.norm(proto_no_v[:,None,:] - proto[None,:, :], dim=2, p=2)
#         actual_proto_no_vdist = torch.norm(actual_proto_no_v[:,None,:] - proto[None,:, :], dim=2, p=2)

#         all_dists_encode_no_v, _encode_no_v = torch.topk(encoded_no_vdist, 10, dim=1, largest=False)
#         all_dists_proto_no_v, _proto_no_v = torch.topk(proto_no_vdist, 10, dim=1, largest=False)
#         all_dists_actual_proto_no_v, _actual_proto_no_v = torch.topk(actual_proto_no_vdist, 10, dim=1, largest=False)


#         dist_matrices = [_proto_no_v, _actual_proto_no_v, _encode_no_v]
#         names = [self.work_dir / f"{self.global_step}_proto_no_vel.gif", self.work_dir / f"{self.global_step}_actual_proto_no_vel.gif", self.work_dir / f"{self.global_step}_encoded_no_vel.gif"]

#         for index_, dist_matrix in enumerate(dist_matrices):
#             filenames=[]
#             for ix, x in enumerate(goal_array):
#                 print('no vel',ix)
#                 txt=''
#                 df = pd.DataFrame()
#                 for i in range(a.shape[0]+1):
#                     if i!=a.shape[0]:
#                         df.loc[i,'x'] = a[i,0]
#                         df.loc[i,'y'] = a[i,1]
#                         if i in dist_matrix[ix,:]:
#                             df.loc[i, 'c'] = 'blue'
#                             z=dist_matrix[ix,(dist_matrix[ix,:] == i).nonzero(as_tuple=True)[0]]
#                             txt += ' ['+str(np.round(state[z][0],2))+','+str(np.round(state[z][1],2))+'] '
#                         else:
#                             df.loc[i,'c'] = 'orange'
#                     else:
#                         df.loc[i,'x'] = x[0].item()
#                         df.loc[i,'y'] = x[1].item()
#                         df.loc[i,'c'] = 'green'


#                 plt.clf()
#                 fig, ax = plt.subplots()
#                 palette = {
#                                     'blue': 'tab:blue',
#                                     'orange': 'tab:orange',
#                                     'green': 'tab:green'
#                                 }
#                 ax=sns.scatterplot(x="x", y="y",
#                           hue="c", palette=palette,
#                           data=df,legend=False)
#                 ax.set_title("\n".join(wrap(txt,75)))
#                 if index_==0:
#                     file1= self.work_dir / f"10nn_proto_goals_no_vel_{ix}_{self.global_step}.png"
#                 elif index_==1:
#                     file1= self.work_dir / f"10nn_actual_proto_goals_no_vel_{ix}_{self.global_step}.png"
                
#                 elif index_==2:
#                     file1= self.work_dir / f"10nn_encoded_no_vel{ix}_{self.global_step}.png"
#                 plt.savefig(file1)
#                 filenames.append(file1)

#             if len(filenames)>100:
#                 filenames=filenames[:100]
#             with imageio.get_writer(os.path.join(self.work_dir,names[index_]), mode='I') as writer:
#                 for file in filenames:
#                     image = imageio.imread(file)
#                     writer.append_data(image)

#             gif = imageio.mimread(os.path.join(self.work_dir ,names[index_]))

#             imageio.mimsave(os.path.join(self.work_dir ,names[index_]), gif, fps=.5)




        #swap goal & rand 1000 samples?
#         encoded_dist = torch.norm(encoded[:,None, :] - encoded[None,:, :], dim=2, p=2)
        proto_dist = torch.norm(self.agent.protos.detach().clone()[:,None,:] - proto[None,:, :], dim=2, p=2)
        all_dists_proto, _proto = torch.topk(proto_dist, 10, dim=1, largest=False)

        with torch.no_grad():
            proto_sim = torch.exp(-1/2*torch.square(self.agent.protos.detach().clone()[:,None,:] - proto[None,:, :]))
        all_dists_proto_sim, _proto_sim = torch.topk(proto_sim, 10, dim=1, largest=True)

        proto_self = torch.norm(self.agent.protos.detach().clone()[:,None,:] - self.agent.protos.detach().clone()[None,:, :], dim=2, p=2)
        all_dists_proto_self, _proto_self = torch.topk(proto_self, self.agent.protos.detach().clone().shape[0], dim=1, largest=False)

        with torch.no_grad():
            proto_sim_self = torch.mm(self.agent.protos.detach().clone(), self.agent.protos.detach().clone().T)
        all_dists_proto_sim_self, _proto_sim_self = torch.topk(proto_sim_self, self.agent.protos.detach().clone().shape[0], dim=1, largest=True)

        dist_matrices = [_proto, _proto_sim]
        self_mat = [_proto_self, _proto_sim_self]
        names = [self.work_dir / f"{self.global_step}_prototypes.gif", self.work_dir / f"{self.global_step}_prototypes_sim.gif"]

        for index_, dist_matrix in enumerate(dist_matrices):
            filenames=[]
            order = self_mat[index_][0,:].cpu().numpy()
            for ix, x in enumerate(order):
                print('proto', ix)
                txt=''
                df = pd.DataFrame()
                for i in range(a.shape[0]+1):
                    if i!=a.shape[0]:
                        df.loc[i,'x'] = a[i,0]
                        df.loc[i,'y'] = a[i,1]
                        df.loc[i,'distance_to_proto1'] = _proto_self[ix,0].item()

                        if i in dist_matrix[x,:]:
                            df.loc[i, 'c'] = 'blue'
                            z=dist_matrix[x,(dist_matrix[x,:] == i).nonzero(as_tuple=True)[0]]
                            txt += ' ['+str(np.round(state[z][0],2))+','+str(np.round(state[z][1],2))+'] '
                        else:
                            df.loc[i,'c'] = 'orange'

                #order based on distance to first prototype

                plt.clf()
                palette = {
                                    'blue': 'tab:blue',
                                    'orange': 'tab:orange'
                                }
                fig, ax = plt.subplots()
                ax=sns.scatterplot(x="x", y="y",
                          hue="c", palette=palette,
                          data=df,legend=False)
                ax.set_title("\n".join(wrap(txt,75)))
                if index_==0:
                    file1= self.work_dir / f"10nn_actual_prototypes_{ix}_{self.global_step}.png"
                elif index_==1:
                    file1= self.work_dir / f"10nn_actual_prototypes_sim_{ix}_{self.global_step}.png"

                plt.savefig(file1)
                filenames.append(file1)

            if len(filenames)>100:
                filenames=filenames[:100]
            with imageio.get_writer(os.path.join(self.work_dir ,names[index_]), mode='I') as writer:
                for file in filenames:
                    image = imageio.imread(file)
                    writer.append_data(image)

            gif = imageio.mimread(os.path.join(self.work_dir ,names[index_]))

            imageio.mimsave(os.path.join(self.work_dir ,names[index_]), gif, fps=.5)
    #######################################################################################
    
    
    def eval_proto(self):
        heatmaps(self, self.global_step)
        eval_env_goal = dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None)


        protos = self.agent.protos.weight.data.detach().clone()

        replay_buffer = make_replay_offline(eval_env_goal,
                                                self.work_dir / 'buffer' / 'buffer_copy',
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
        state = state.reshape((state.shape[0],4))

        num_sample=600 
        idx = np.random.randint(0, state.shape[0], size=num_sample)
        state=state[idx]
        state=state.reshape(num_sample,4)
        a = state
        count10,count01,count00,count11=(0,0,0,0)
        # density estimate:
        df = pd.DataFrame()
        for state_ in a:
            if state_[0]<0:
                if state_[1]>=0:
                    count10+=1
                else:
                    count00+=1
            else:
                if state_[1]>=0:
                    count11+=1
                else:
                    count01+=1

        df.loc[0,0] = count00/a.shape[0]
        df.loc[0,1] = count01/a.shape[0]
        df.loc[1,1] = count11/a.shape[0]
        df.loc[1,0] = count10/a.shape[0]
        labels=df
        plt.clf()
        fig, ax = plt.subplots()
        sns.heatmap(df, cmap="Blues_r",cbar=False, annot=labels).invert_yaxis()
        ax.set_title('data percentage')
        plt.savefig(self.work_dir / f"data_pct_model_{self._global_step}.png")

        def ndim_grid(ndims, space):
            L = [np.linspace(-.25,.25,space) for i in range(ndims)]
            return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T

        lst=[]
        goal_array = ndim_grid(2,10)
        for ix,x in enumerate(goal_array):
            if (-.2<x[0]<.2 and -.02<x[1]<.02) or (-.02<x[0]<.02 and -.2<x[1]<.2):
                lst.append(ix)


        goal_array=np.delete(goal_array, lst,0)
        emp = np.zeros((goal_array.shape[0],2))
        goal_array = np.concatenate((goal_array, emp), axis=1)


        #pixels = []
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



        #swap goal & rand 1000 samples?
#         encoded_dist = torch.norm(encoded[:,None, :] - encoded[None,:, :], dim=2, p=2)
        proto_dist = torch.norm(protos[:,None,:] - proto[None,:, :], dim=2, p=2)
        all_dists_proto, _proto = torch.topk(proto_dist, 10, dim=1, largest=False)

        with torch.no_grad():
            #proto_sim = self.agent.protos(proto).T
            proto_sim = torch.exp(-1/2*torch.square(torch.norm(protos[:,None,:] - proto[None,:, :], dim=2, p=2)))
        all_dists_proto_sim, _proto_sim = torch.topk(proto_sim, 10, dim=1, largest=True)

        proto_self = torch.norm(protos[:,None,:] - protos[None,:, :], dim=2, p=2)
        all_dists_proto_self, _proto_self = torch.topk(proto_self, protos.shape[0], dim=1, largest=False)

        with torch.no_grad():
            proto_sim_self = self.agent.protos(protos).T
        all_dists_proto_sim_self, _proto_sim_self = torch.topk(proto_sim_self, protos.shape[0], dim=1, largest=True)
        
        dist_matrices = [_proto, _proto_sim]
        self_mat = [_proto_self, _proto_sim_self]
        names = [self.work_dir / f"{self.global_step}_prototypes.gif", self.work_dir / f"{self.global_step}_prototypes_sim.gif"]

        for index_, dist_matrix in enumerate(dist_matrices):
            filenames=[]
            order = self_mat[index_][0,:].cpu().numpy()
            plt.clf()
            fig, ax = plt.subplots()
            dist_np = np.empty((_proto_self.shape[1], dist_matrix.shape[1], 2))
            
            for ix, x in enumerate(order):
                txt=''
                df = pd.DataFrame()
                count=0
                for i in range(a.shape[0]+1):
                    if i!=a.shape[0]:
                        df.loc[i,'x'] = a[i,0]
                        df.loc[i,'y'] = a[i,1]
                        df.loc[i,'distance_to_proto1'] = _proto_self[ix,0].item()
                        
                        if i in dist_matrix[ix,:]:
                            df.loc[i, 'c'] = str(ix+1)
                            dist_np[ix,count,0] = a[i,0]
                            dist_np[ix,count,1] = a[i,1]
                            count+=1
                            #z=dist_matrix[ix,(dist_matrix[ix,:] == i).nonzero(as_tuple=True)[0]]
                            #txt += ' ['+str(np.round(state[z][0],2))+','+str(np.round(state[z][1],2))+'] '
                        elif ix==0 and (i not in dist_matrix[ix,:]):
                            #color all samples blue
                            df.loc[i,'c'] = str(0)

                #order based on distance to first prototype
                #plt.clf()
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
                #fig, ax = plt.subplots()
                ax=sns.scatterplot(x="x", y="y",
                          hue="c",palette=palette,
                          data=df,legend=True)
                sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
                #ax.set_title("\n".join(wrap(txt,75)))
            
            pairwise_dist = np.linalg.norm(dist_np[:,0,None,:]-dist_np, ord=2, axis=2)
            print(pairwise_dist.shape)
            
            #maximum pairwise distance amongst prototypes
            maximum = np.amax(pairwise_dist, axis=1)
            num = self.global_step//self.cfg.eval_every_frames
            self.final_df.loc[num, 'avg'] = np.mean(maximum)
            self.final_df.loc[num, 'med'] = np.median(maximum)
            self.final_df.loc[num, 'q9'] = np.quantile(maximum, .9)
            self.final_df.loc[num, 'q8'] = np.quantile(maximum, .8)
            self.final_df.loc[num, 'q7'] = np.quantile(maximum, .7)
            self.final_df.loc[num, 'max'] = np.max(maximum)
            


            if index_==0:
                file1= self.work_dir / f"10nn_actual_prototypes_{self.global_step}.png"
                plt.savefig(file1)
                wandb.save(f"10nn_actual_prototypes_{self.global_step}.png")
            elif index_==1:
                file1= self.work_dir / f"10nn_actual_prototypes_sim_{self.global_step}.png"
                plt.savefig(file1)
                wandb.save(f"10nn_actual_prototypes_sim_{self.global_step}.png")

            #import IPython as ipy; ipy.embed(colors='neutral')
            if self.global_step >= (self.cfg.num_train_frames//2-100):
                fig, ax = plt.subplots()
                self.final_df.plot(ax=ax)
                ax.set_xticks(self.final_df.index)
                plt.savefig(self.work_dir / f"proto_states.png")
                wandb.save(f"proto_states.png") 


            #filenames.append(file1)

            #if len(filenames)>100:
            #    filenames=filenames[:100]
            #with imageio.get_writer(os.path.join(self.work_dir ,names[index_]), mode='I') as writer:
            #    for file in filenames:
            #        image = imageio.imread(file)
            #        writer.append_data(image)
#
#            gif = imageio.mimread(os.path.join(self.work_dir ,names[index_]))

#            imageio.mimsave(os.path.join(self.work_dir ,names[index_]), gif, fps=.5)
    #######################################################################################

    def eval(self):
        heatmap = self.replay_storage.state_visitation_proto

        plt.clf()
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        ax.set_title(self.global_step)

        plt.savefig(f"./{self.global_step}_proto_heatmap.png")
        wandb.save(f"./{self.global_step}_proto_heatmap.png")


        heatmap_pct = self.replay_storage.state_visitation_proto_pct

        plt.clf()
        fig, ax = plt.subplots(figsize=(10,10))
        labels = np.round(heatmap_pct.T/heatmap_pct.sum()*100, 1)
        sns.heatmap(np.log(1 + heatmap_pct.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
        ax.set_title(self.global_step)

        plt.savefig(f"./{self.global_step}_proto_heatmap_pct.png")
        wandb.save(f"./{self.global_step}_proto_heatmap_pct.png")
 
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            goal = np.random.sample((2,)) * .5 - .25
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    if self.cfg.goal:
                        if self.cfg.obs_type=='pixels':
                            action = self.agent.act(time_step.observation['pixels'],
                                            goal,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                        else:
                            action = self.agent.act(time_step.observation,
                                            goal,
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
        meta = self.agent.init_meta()
        if self.cfg.obs_type=='pixels':
            self.replay_storage.add(time_step, meta, True)
        else:
            self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):
            if time_step.last():
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

                # reset env
                if self.cfg.const_init==False:
                    task = PRIMAL_TASKS[self.cfg.domain]
                    rand_init = np.random.uniform(.02,.29,size=(2,))
                    sign = np.array([[1,1],[-1,1],[1,-1],[-1,-1]])
                    rand = np.random.randint(4)
                    self.train_env = dmc.make(self.cfg.task_no_goal, self.cfg.obs_type, self.cfg.frame_stack,
                                                              self.cfg.action_repeat, self.cfg.seed, init_state=(rand_init[0]*sign[rand][0], rand_init[1]*sign[rand][1]))
                    print('sampled init', (rand_init[0]*sign[rand][0], rand_init[1]*sign[rand][1]))   
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                if self.cfg.obs_type=='pixels':

                    self.replay_storage.add(time_step, meta, True)
                else: 
                    self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step) and self.global_step!=0:
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                if self.cfg.debug:
                    self.eval()
                elif self.cfg.agent.name=='protov2':
                    self.eval_protov2()
                else:
                    self.eval_proto()
            
            meta = self.agent.update_meta(meta, self.global_step, time_step)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                if self.cfg.goal:
                    if self.cfg.obs_type=='pixels':
                        action = self.agent.act(time_step.observation['pixels'],
                                            goal,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                    else:    
                        action = self.agent.act(time_step.observation,
                                            goal,
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
                metrics = self.agent.update(self.replay_iter, self.global_step, test=self.cfg.test)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')
            
            #save agent
            if self._global_step%200000==0 and self._global_step!=0:
                path = os.path.join(self.work_dir, 'optimizer_{}_{}.pth'.format(str(self.cfg.agent.name),self._global_step))
                torch.save(self.agent, path)
                
            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            if  self.cfg.obs_type=='pixels':
                self.replay_storage.add(time_step, meta, True)
            else:
                self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
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
    from pretrain import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()

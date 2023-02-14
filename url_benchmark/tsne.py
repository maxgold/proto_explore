import scipy
import glob
import seaborn as sns
import pandas as pd
import re
import natsort
import random
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import imageio
import warnings
from textwrap import wrap

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
import io
from sklearn.manifold import TSNE
import natsort


torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS
import time

#models = ['/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/exp_local/2023.02.08/192418_ddpg_only/', '/vast/nm1874/dm_control_2022/proto_explore/url_benchmark/exp_local/2023.02.08/231828_ddpg_only/']
models=['/home/nina/proto_explore/url_benchmark/exp_local/2023.02.09/230853_proto_encoder2/', '/home/nina/proto_explore/url_benchmark/exp_local/2023.02.09/230849_proto_encoder2/', '/home/nina/proto_explore/url_benchmark/exp_local/2023.02.09/230900_proto_encoder3/', '/home/nina/proto_explore/url_benchmark/exp_local/2023.02.09/230856_proto_encoder3/']


for m in models:
    model = m.split('/')[-3] + '_' +m.split('/')[-2]
    tmp_agent_name = m.split('/')[-2].split('_')
    print(tmp_agent_name)
    agent_name = tmp_agent_name[-2] + '_' + tmp_agent_name[-1]
    paths = natsort.natsorted(glob.glob(m+'*00000.pth'))[-1]
    print(paths)
    paths=[paths]
    for path in paths:
        model_step = path.split('_')[-1].split('.')[0]
        print('model', m)
        print('model step', model_step)
        if model_step=='0':
            continue
        print(path)
        agent  = torch.load(path,map_location='cuda:0')
        eval_env_goal = dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None)
        env = dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None)
        
        
        replay_dir = Path(m+'buffer2/buffer_copy/')
        

        def ndim_grid(ndims, space):
            L = [np.linspace(-.29,.29,space) for i in range(ndims)]
            return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T

        lst=[]
        goal_array = ndim_grid(2,10)
        for ix,x in enumerate(goal_array):
            if (-.2<x[0]<.2 and -.02<x[1]<.02) or (-.02<x[0]<.02 and -.2<x[1]<.2):
                lst.append(ix)

        
        goal_array=np.delete(goal_array, lst,0)
        emp = np.zeros((goal_array.shape[0],2))
        goal_array = np.concatenate((goal_array, emp), axis=1)

        plt.clf()
        fig, ax = plt.subplots()
        ax.scatter(goal_array[:,0], goal_array[:,1])
        plt.savefig(f"./tsne_output/mesh.png")
        lst=[]
        goal_array = torch.as_tensor(goal_array, device=torch.device('cuda:0'))


        encoded_no_v=[]
        goal_array = ndim_grid(2,10)
        for ix,x in enumerate(goal_array):
            if (-.2<x[0]<.2 and -.02<x[1]<.02) or (-.02<x[0]<.02 and -.2<x[1]<.2):
                lst.append(ix)
        goal_array=np.delete(goal_array, lst,0)
        print('goal', goal_array.shape)
        for x in goal_array:
            with torch.no_grad():
                with eval_env_goal.physics.reset_context():
                    time_step_init = eval_env_goal.physics.set_state(np.array([x[0].item(), x[1].item(),0,0]))

                time_step_init = eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
                time_step_init = np.transpose(time_step_init, (2,0,1))
                time_step_init = np.tile(time_step_init, (3,1,1))

                obs = time_step_init
                obs = torch.as_tensor(obs.copy(), device=torch.device('cuda:0')).unsqueeze(0)
                
                z = agent.encoder(obs)
                z = torch.cat([z,z], dim=-1)
                z = agent.actor.trunk(z) 
                encoded_no_v.append(z)

        print('encoder', encoded_no_v[0].shape)
        encoded_no_v = torch.cat(encoded_no_v,axis=0)
        
        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=300)
        tsne_results = tsne.fit_transform(encoded_no_v.cpu().numpy())
        
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
        df_subset=pd.DataFrame() 
        df_subset['tsne-2d-one'] = tsne_results[:,0]
        df_subset['tsne-2d-two'] = tsne_results[:,1]
        for ix, x in enumerate(goal_array):
            if x[0] < -.02 and x[1] > .02:
                df_subset.loc[ix, 'y'] = 1
            elif x[0] > .02 and x[1] > .02:
                df_subset.loc[ix, 'y'] = 2
            elif x[0] > .02 and x[1] < -.02:
                df_subset.loc[ix, 'y'] = 3
            elif x[0] < -.02 and x[1] < -.02:
                df_subset.loc[ix, 'y'] = 4
        
        
            
        plt.clf()
        fig, ax = plt.subplots(figsize=(16,10))
        ax = sns.scatterplot(
                x="tsne-2d-one", y="tsne-2d-two",
                hue='y',
                palette=sns.color_palette("hls", 4),
                data=df_subset,
                legend="full",
                alpha=1
                    )
        plt.savefig(f"./tsne_output/tsne_grid_model{model}_{model_step}.png")

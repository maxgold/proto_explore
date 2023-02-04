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


torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS
import time

#models = ['/home/nina/proto_explore/url_benchmark/exp_local/2022.10.21/151650_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.20/231842_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.20/231819_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.20/231715_proto_encoder1/']

models = ['/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2023.01.27/234846_proto_encoder1/']
#models = ['/home/ubuntu/proto_explore/url_benchmark/exp_local/2022.10.14/210339_proto_encoder1/']
#models = ['/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.10.12/215650_proto_encoder3/', '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.10.12/215751_proto_encoder3/']
#models = ['/home/ubuntu/proto_explore/url_benchmark/exp_local/2022.09.09/072830_proto/']

#models = ['/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.10.14/010502_proto_encoder1/','/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.10.12/215650_proto_encoder3/', '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.10.12/215751_proto_encoder3/','/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.10.10/213447_proto_encoder0/', '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.10.10/213328_proto_encoder2/', '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.10.10/213411_proto_encoder1/', '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.10.09/231012_proto_encoder2/', '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.10.09/203156_proto_encoder0/']


for m in models:
    model = m.split('/')[-3] + '_' +m.split('/')[-2]
    tmp_agent_name = m.split('/')[-2].split('_')
    print(tmp_agent_name)
    agent_name = tmp_agent_name[-2] + '_' + tmp_agent_name[-1]
    paths = glob.glob(m+'*00000.pth')
    print(paths)
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
        
        protos = agent.protos.weight.data.detach().clone()
        
        replay_dir = Path(m+'buffer2/buffer_copy/')
        
        replay_buffer = make_replay_offline(eval_env_goal,
                                                replay_dir,
                                                100000,
                                                0,
                                                0,
                                                .99,
                                                goal=False,
                                                relabel=False,
                                                model_step = int(model_step),
                                                replay_dir2=False,
                                                obs_type = 'pixels'
                                                )

        state, actions, rewards, eps, index = replay_buffer.parse_dataset() 
        state = state.reshape((state.shape[0],4))
        print(state.shape)
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
            state_t[ix] = state[x]
            fn = eps[x]
            idx_ = index[x]
            ep = np.load(fn)
            #pixels.append(ep['observation'][idx_])

            with torch.no_grad():
                obs = ep['observation'][idx_]
                obs = torch.as_tensor(obs.copy(), device=torch.device('cuda:0'))
                z = agent.encoder(obs)
                encoded.append(z)
                z = agent.predictor(z)
                z = agent.projector(z)
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
        plt.savefig(f"./tsne_output/singular_value_{model}_{model_step}.png")
           
        
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
                    
        df.loc[0,0] = count00
        df.loc[0,1] = count01
        df.loc[1,1] = count11
        df.loc[1,0] = count10
        labels=df
        plt.clf()
        fig, ax = plt.subplots()
        sns.heatmap(df, cmap="Blues_r",cbar=False, annot=labels).invert_yaxis()
        ax.set_title('data percentage')
        plt.savefig(f"./tsne_output/data_pct_model{model}_{model_step}.png")

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

        plt.clf()
        fig, ax = plt.subplots()
        ax.scatter(goal_array[:,0], goal_array[:,1])
        plt.savefig(f"./tsne_output/mesh.png")
        lst=[]
        goal_array = torch.as_tensor(goal_array, device=torch.device('cuda:0'))

        plt.clf()
        fig, ax = plt.subplots()
        ax.scatter(a[:,0], a[:,1])
        plt.savefig(f"./tsne_output/samples.png")
        a = torch.as_tensor(a,device=torch.device('cuda:0'))

        state_dist = torch.norm(goal_array[:,None,:]  - a[None,:,:], dim=2, p=2)
        all_dists_state, _state = torch.topk(state_dist, 10, dim=1, largest=False)

        test_states = np.array([[-.15, -.15], [-.15, .15], [.15, -.15], [.15, .15]])
        action = np.array([[.5, 0], [-.5, 0],[0, .5], [0, -.5]])
 

        ##encoded goals w/ no velocity 

        actual_proto_no_v=[]
        encoded_no_v=[]
        proto_no_v = []
        #no velocity goals 
        actual_proto_no_v = []
        goal_array = ndim_grid(2,10)
        for ix,x in enumerate(goal_array):
            if (-.2<x[0]<.2 and -.02<x[1]<.02) or (-.02<x[0]<.02 and -.2<x[1]<.2):
                lst.append(ix)
        goal_array=np.delete(goal_array, lst,0)

        lst_proto = []
        for x in goal_array:
            with torch.no_grad():
                with eval_env_goal.physics.reset_context():
                    time_step_init = eval_env_goal.physics.set_state(np.array([x[0].item(), x[1].item(),0,0]))

                time_step_init = eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
                time_step_init = np.transpose(time_step_init, (2,0,1))
                #time_step_init = np.tile(time_step_init, (3,1,1))

                obs = time_step_init
                obs = torch.as_tensor(obs.copy(), device=torch.device('cuda:0')).unsqueeze(0)
                z = agent.encoder(obs)
                encoded_no_v.append(z)
                z = agent.predictor(z)
                z = agent.projector(z)
                z = F.normalize(z, dim=1, p=2)
                proto_no_v.append(z)
                sim = agent.protos(z)
                idx_ = sim.argmax()
                lst_proto.append(idx_)
                actual_proto_no_v.append(protos[idx_][None,:])

        print('ndim_grid no velocity: therere {} unique prototypes that are neighbors to {} datapoints'.format(len(set(lst_proto)), goal_array.shape[0]))


        encoded_no_v = torch.cat(encoded_no_v,axis=0)
        proto_no_v = torch.cat(proto_no_v,axis=0)
        actual_proto_no_v = torch.cat(actual_proto_no_v,axis=0)
        
        time_start = time.time()
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
        tsne_results = tsne.fit_transform(encoded_no_v.cpu().numpy())
        
        print('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))
        df_subset=pd.DataFrame() 
        df_subset['tsne-2d-one'] = tsne_results[:,0]
        df_subset['tsne-2d-two'] = tsne_results[:,1]
        for ix, x in enumerate(goal_array):
            if x[0] < .02 and x[1] > -.04:
                df_subset.loc[ix, 'y'] = 1
            elif x[0] >= .02 and x[1] > -.04:
                df_subset.loc[ix, 'y'] = 2
            elif x[0] >= .02 and x[1] < -.04:
                df_subset.loc[ix, 'y'] = 3
            else:
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

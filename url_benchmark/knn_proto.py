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


torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS

models = ['/home/nina/proto_explore/url_benchmark/exp_local/2022.10.22/114432_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.22/113750_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.22/113741_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.21/151650_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.21/151635_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.21/151501_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.21/151446_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.21/151431_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.21/151414_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.20/231842_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.20/231819_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.20/231802_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.20/231715_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.20/231631_proto_encoder1/', '/home/nina/proto_explore/url_benchmark/exp_local/2022.10.20/231602_proto_encoder1/']
#models = ['/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.09.21/150106_proto/', '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.09.21/150022_proto/', '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.09.21/145803_proto/']

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
    for path in paths:
        model_step = path.split('_')[-1].split('.')[0]
        print('model', m)
        print('model step', model_step)
        if model_step=='0':
            continue
        print(path)
        agent  = torch.load(path,map_location='cuda:3')
        eval_env_goal = dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None)
        env = dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None)
        
        protos = agent.protos.weight.data.detach().clone()
        
        if model == '2022.09.09_072830_proto':
            replay_dir = Path(m+'buffer2/buffer_copy/')
        else:
            replay_dir = Path(m+'buffer2/buffer_copy/')

        replay_buffer = make_replay_offline(eval_env_goal,
                                                replay_dir,
                                                100000,
                                                0,
                                                0,
                                                .99,
                                                goal=False,
                                                relabel=False,
                                                model_step = model_step,
                                                replay_dir2=False,
                                                obs_type = 'pixels'
                                                )

        state, actions, rewards, eps, index = replay_buffer.parse_dataset() 
        state = state.reshape((state.shape[0],4))
        print(state.shape)
       
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
        plt.savefig(f"./knn_proto/data_pct_model{model}_{model_step}.png")

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
                obs = torch.as_tensor(obs.copy(), device=torch.device('cuda:3')).unsqueeze(0)
                z = agent.encoder(obs)
                encoded.append(z)
                z = agent.predictor(z)
                z = agent.projector(z)
                z = F.normalize(z, dim=1, p=2)
                proto.append(z)
                sim = agent.protos(z)
                idx_ = sim.argmax()
                actual_proto.append(protos[idx_][None,:])


        encoded = torch.cat(encoded,axis=0)
        proto = torch.cat(proto,axis=0)
        actual_proto = torch.cat(actual_proto,axis=0)
        
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
                time_step_init = np.tile(time_step_init, (3,1,1))

                obs = time_step_init
                obs = torch.as_tensor(obs, device=torch.device('cuda:3')).unsqueeze(0)
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
        
                #no velocity goals 


        encoded_no_vdist = torch.norm(encoded_no_v[:,None, :] - encoded[None,:, :], dim=2, p=2)
        proto_no_vdist = torch.norm(proto_no_v[:,None,:] - proto[None,:, :], dim=2, p=2)
        actual_proto_no_vdist = torch.norm(actual_proto_no_v[:,None,:] - proto[None,:, :], dim=2, p=2)

        all_dists_encode_no_v, _encode_no_v = torch.topk(encoded_no_vdist, 10, dim=1, largest=False)
        all_dists_proto_no_v, _proto_no_v = torch.topk(proto_no_vdist, 10, dim=1, largest=False)
        all_dists_actual_proto_no_v, _actual_proto_no_v = torch.topk(actual_proto_no_vdist, 10, dim=1, largest=False)

        with torch.no_grad():
            proto_no_v_sim = agent.protos(proto_no_v)
            actual_proto_no_v_sim = agent.protos(actual_proto_no_v)
        all_dists_proto_no_v_sim, _proto_no_v_sim = torch.topk(proto_no_v_sim, 10, dim=1, largest=True)
        actual_all_dists_proto_no_v_sim, _actual_proto_no_v_sim = torch.topk(actual_proto_no_v_sim, 10, dim=1, largest=True)


        dist_matrices = [_proto_no_v, _actual_proto_no_v, _proto_no_v_sim, _actual_proto_no_v_sim]
        names = [f"{model}_{model_step}_proto_no_vel.gif", f"{model}_{model_step}_actual_proto_no_vel.gif", f"{model}_{model_step}_sim_proto_no_vel.gif", f"{model}_{model_step}_sim_actual_proto_no_vel.gif", f"{model}_{model_step}_encoded_no_vel.gif"]

        for index_, dist_matrix in enumerate(dist_matrices):
            filenames=[]
            for ix, x in enumerate(goal_array):
                print('no vel',ix)
                txt=''
                df = pd.DataFrame()
                for i in range(a.shape[0]+1):
                    if i!=a.shape[0]:
                        df.loc[i,'x'] = a[i,0]
                        df.loc[i,'y'] = a[i,1]
                        if i in dist_matrix[ix,:]:
                            df.loc[i, 'c'] = 'blue'
                            z=dist_matrix[ix,(dist_matrix[ix,:] == i).nonzero(as_tuple=True)[0]]
                            txt += ' ['+str(np.round(state[z][0],2))+','+str(np.round(state[z][1],2))+'] '
                        else:
                            df.loc[i,'c'] = 'orange'
                    else:
                        df.loc[i,'x'] = x[0].item()
                        df.loc[i,'y'] = x[1].item()
                        df.loc[i,'c'] = 'green'


                plt.clf()
                fig, ax = plt.subplots()
                palette = {
                                    'blue': 'tab:blue',
                                    'orange': 'tab:orange',
                                    'green': 'tab:green'
                                }
                ax=sns.scatterplot(x="x", y="y",
                          hue="c", palette=palette,
                          data=df,legend=False)
                ax.set_title("\n".join(wrap(txt,75)))
                if index_==0:
                    file1= f"./knn_proto/10nn_proto_goals_no_vel_{ix}_model{model}_{model_step}.png"
                elif index_==1:
                    file1= f"./knn_proto/10nn_actual_proto_goals_no_vel_{ix}_model{model}_{model_step}.png"
                elif index_==2:
                    file1= f"./knn_proto/10nn_sim_proto_goals_no_vel{ix}_model{model}_{model_step}.png"
                elif index_==3:
                    file1= f"./knn_proto/10nn_sim_actual_proto_goals_no_vel{ix}_model{model}_{model_step}.png"
                elif index_==4:
                    file1= f"./knn_proto/10nn_encoded_no_vel{ix}_model{model}_{model_step}.png"
                plt.savefig(file1)
                filenames.append(file1)

            if len(filenames)>100:
                filenames=filenames[:100]
            with imageio.get_writer(os.path.join('./knn_output/',names[index_]), mode='I') as writer:
                for file in filenames:
                    image = imageio.imread(file)
                    writer.append_data(image)

            gif = imageio.mimread(os.path.join('./knn_output/',names[index_]))

            imageio.mimsave(os.path.join('./knn_proto/',names[index_]), gif, fps=.5)
            

        #swap goal & rand 1000 samples?

        proto_dist = torch.norm(protos[:,None,:] - proto[None,:, :], dim=2, p=2)
        all_dists_proto, _proto = torch.topk(proto_dist, 10, dim=1, largest=False)

        with torch.no_grad():
            proto_sim = agent.protos(proto).T
        all_dists_proto_sim, _proto_sim = torch.topk(proto_sim, 10, dim=1, largest=True)

        proto_self = torch.norm(protos[:,None,:] - protos[None,:, :], dim=2, p=2)
        all_dists_proto_self, _proto_self = torch.topk(proto_self, protos.shape[0], dim=1, largest=False)

        with torch.no_grad():
            proto_sim_self = agent.protos(protos).T
        all_dists_proto_sim_self, _proto_sim_self = torch.topk(proto_sim_self, protos.shape[0], dim=1, largest=True)

        dist_matrices = [_proto, _proto_sim]
        self_mat = [_proto_self, _proto_sim_self]
        names = [f"{model}_{model_step}_prototypes.gif", f"{model}_{model_step}_prototypes_sim.gif"]
        filenames=[]
        
        for index_, dist_matrix in enumerate(dist_matrices):
            order = self_mat[index_][0,:].cpu().numpy()
            plt.clf()
            fig, ax = plt.subplots()
            for ix in range(min(_proto_self.shape[1], 32)):
                print('proto', ix)
                txt=''
                df = pd.DataFrame()
                for i in range(a.shape[0]+1):
                    if i!=a.shape[0]:
                        df.loc[i,'x'] = a[i,0]
                        df.loc[i,'y'] = a[i,1]
                        df.loc[i,'distance_to_proto1'] = _proto_self[ix,0].item()

                        if i in dist_matrix[ix,:]:
                            df.loc[i, 'c'] = str(ix+1)
                            z=dist_matrix[ix,(dist_matrix[ix,:] == i).nonzero(as_tuple=True)[0]]
                            #txt += ' ['+str(np.round(state[z][0],2))+','+str(np.round(state[z][1],2))+'] '
                        elif ix==0 and (i not in dist_matrix[ix,:]):
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
                          data=df,legend=False)
                #ax.set_title("\n".join(wrap(txt,75)))
                
            if index_==0:
                file1= f"./knn_proto/10nn_actual_prototypes_model{model}_{model_step}.png"
            elif index_==1:
                file1= f"./knn_proto/10nn_actual_prototypes_sim_model{model}_{model_step}.png"
                
            filenames.append(file1)
            plt.savefig(file1)
            
        if len(filenames)>100:
            filenames=filenames[:100]
            
        with imageio.get_writer(os.path.join('./knn_proto/',names[index_]), mode='I') as writer:
            
            for file in filenames:
                
                image = imageio.imread(file)
                writer.append_data(image)

        gif = imageio.mimread(os.path.join('./knn_proto/',names[index_]))

        imageio.mimsave(os.path.join('./knn_proto/',names[index_]), gif, fps=.5)
            

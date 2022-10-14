
import glob
import seaborn as sns
import pandas as pd
import re
import natsort
import random
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import imageio.v2 as imageio
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

models = ['/home/ubuntu/proto_explore/url_benchmark/exp_local/2022.09.09/072830_proto/']
#models = ['/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.10.10/213447_proto_encoder0/', '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.10.10/213328_proto_encoder2/', '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.10.10/213411_proto_encoder1/', '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.10.09/231012_proto_encoder2/', '/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.10.09/203156_proto_encoder0/']

for m in models:
    model = m.split('/')[-3] + '_' +m.split('/')[-2]
    model_name = model.replace('.', '_')
    
    agent  = torch.load(m+'optimizer_proto_1000000.pth')

    eval_env_goal = dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None)
    env = dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None)
    
    if model == '2022.09.09_072830_proto':
        replay_dir = Path(m+'buffer2/buffer_copy/')
    else:
        replay_dir = Path(m+'buffer/buffer_copy/')

    replay_buffer = make_replay_offline(eval_env_goal,
                                            replay_dir,
                                            100000,
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
    idx = np.random.randint(0, state.shape[0], size=380)
    state=state[idx]
    state=state.reshape(380,4)
    a = state
    #fn = Path('./knn_visual/samples.npz')
    #with io.BytesIO() as bs:
    #    np.savez_compressed(bs, a)
    #    bs.seek(0)
    #    with fn.open("wb") as f:
    #        f.write(bs.read())


    def ndim_grid(ndims, space):
        L = [np.linspace(-.25,.25,space) for i in range(ndims)]
        return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T

    lst=[]
    goal_array = ndim_grid(2,10)
    for ix,x in enumerate(goal_array):
        if (-.2<x[0]<.2 and -.02<x[1]<.02) or (-.02<x[0]<.02 and -.2<x[1]<.2):
            lst.append(ix)

    protos = agent.protos.weight.data.detach().clone()
    goal_array=np.delete(goal_array, lst,0)
    emp = np.zeros((goal_array.shape[0],2))
    goal_array = np.concatenate((goal_array, emp), axis=1)

    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(goal_array[:,0], goal_array[:,1])
    plt.savefig(f"./knn_output/mesh.png")
    lst=[]
    goal_array = torch.as_tensor(goal_array, device=torch.device('cuda'))

    plt.clf()
    fig, ax = plt.subplots()
    ax.scatter(a[:,0], a[:,1])
    plt.savefig(f"./knn_output/samples.png")
    a = torch.as_tensor(a,device=torch.device('cuda'))

    state_dist = torch.norm(goal_array[:,None,:]  - a[None,:,:], dim=2, p=2)
    all_dists_state, _state = torch.topk(state_dist, 10, dim=1, largest=False)

    test_states = np.array([[-.15, -.15], [-.15, .15], [.15, -.15], [.15, .15]])
    action = np.array([[.5, 0], [-.5, 0],[0, .5], [0, -.5]])

    ###############################################################################################################################
    #.0002
    #velocity test 

    count=0
    goal_w_vel = torch.empty((16, 9,84,84))
    goalwvel = []
    for x in test_states:
        for iy, y in enumerate(action):
            tmp_goal = []
            with torch.no_grad():
                with eval_env_goal.physics.reset_context():
                    if iy==0:
                        time_step_init = eval_env_goal.physics.set_state(np.array([x[0].item()-.0002, x[1].item(),0,0]))
                    elif iy==1:
                        time_step_init = eval_env_goal.physics.set_state(np.array([x[0].item()+.0002, x[1].item(),0,0]))
                    elif iy==2:
                        time_step_init = eval_env_goal.physics.set_state(np.array([x[0].item(), x[1].item()-.0002,0,0]))
                    elif iy==3:
                        time_step_init = eval_env_goal.physics.set_state(np.array([x[0].item(), x[1].item()+.0002,0,0]))

                time_step_init = eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
                time_step_init = np.transpose(time_step_init, (2,0,1))
                time_step_init = torch.as_tensor(time_step_init.copy())
                tmp_goal.append(time_step_init)

            for i in range(3):

                #time_step_init = eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
                #time_step_init = np.transpose(time_step_init, (2,0,1))

                if i==0:
                    if iy==0:
                        env=dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None, init_state=(x[0].item()-.0002, x[1].item()))
                        time_step = env.reset()
                    elif iy==1:
                        env=dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None, init_state=(x[0].item()+.0002, x[1].item()))
                        time_step = env.reset()

                    elif iy==2:
                        env=dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None, init_state=(x[0].item(), x[1].item()-.0002))
                        time_step = env.reset()
                    elif iy==3:
                        env=dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None, init_state=(x[0].item(), x[1].item()+.0002))
                        time_step = env.reset()
                else:

                    current = time_step.observation['observations']
                    with torch.no_grad(): 
                        with eval_env_goal.physics.reset_context():

                            time_step_init = eval_env_goal.physics.set_state(np.array([current[0], current[1],0,0]))
                        time_step_init = eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
                        time_step_init = np.transpose(time_step_init, (2,0,1))
                        time_step_init = torch.as_tensor(time_step_init.copy())
                        tmp_goal.append(time_step_init)


                    env=dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None, init_state=current[:2])
                    time_step = env.reset()

                time_step = env.step(y)

            tmp_goal = torch.cat(tmp_goal, axis=0)
            goal_w_vel[count] = tmp_goal
            goalwvel.append(tmp_goal)
            count+=1

    # final_encoded = np.array([i.detach().clone().cpu().numpy() for i in goal_w_vel])
    # fn = [Path('./test_frames.npz')]
    # for ix,x in enumerate(fn):
    #     with io.BytesIO() as bs:
    #         np.savez_compressed(bs, final_encoded)
    #         bs.seek(0)
    #         with x.open("wb") as f:
    #             f.write(bs.read())

    encoded_v=[]
    proto_v = []
    lst_proto = []
    actual_proto_v = []
    for x in range(16):
        with torch.no_grad(): 
            obs = goal_w_vel[x]
            obs = torch.as_tensor(obs, device=torch.device('cuda')).unsqueeze(0)
            z = agent.encoder(obs)
            encoded_v.append(z)
            z = agent.predictor(z)
            #z = agent.projector(z)
            z = F.normalize(z, dim=1, p=2)
            proto_v.append(z)
            sim = agent.protos(z)
            idx_ = sim.argmax()
            lst_proto.append(idx_)
            actual_proto_v.append(protos[idx_][None,:])
    
    print('test states w/ velocity: therere {} unique prototypes that are neighbors to {} datapoints'.format(len(set(lst_proto)), 16))

    encoded_v = torch.cat(encoded_v,axis=0)
    proto_v = torch.cat(proto_v,axis=0)
    actual_proto_v = torch.cat(actual_proto_v,axis=0)



    # vel_encode_dist = torch.norm(encoded_v[:,None, :] - encoded_v[None,:, :], dim=2, p=2)
    vel_proto_dist = torch.norm(proto_v[:,None,:] - proto_v[None,:, :], dim=2, p=2)
    # all_dists_encode_v, _encode_v = torch.topk(vel_encode_dist, 16, dim=1, largest=False)
    all_dists_proto_v, _proto_v = torch.topk(vel_proto_dist, 16, dim=1, largest=False)

    print(all_dists_proto_v)
#     df = pd.DataFrame(vel_encode_dist.cpu().numpy())
#     df.to_csv(f'./knn_output/encode_dist_{model}.csv', index=False)
    df = pd.DataFrame(vel_proto_dist.cpu().numpy())
    df.to_csv(f'./knn_output/proto_dist_{model}.csv', index=False)

    ##########################################################################################################################        

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
            obs = torch.as_tensor(obs, device=torch.device('cuda')).unsqueeze(0)
            z = agent.encoder(obs)
            encoded_no_v.append(z)
            z = agent.predictor(z)
            #z = agent.projector(z)
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

    pixels = []
    encoded = []
    proto = []
    actual_proto = []
    lst_proto = []

    for x in idx:
        fn = eps[x]
        idx_ = index[x]
        ep = np.load(fn)
        pixels.append(ep['observation'][idx_])

    for x in pixels:
        with torch.no_grad():
            obs = x
            obs = torch.as_tensor(obs.copy(), device=torch.device('cuda')).unsqueeze(0)
            z = agent.encoder(obs)
            encoded.append(z)
            z = agent.predictor(z)
            #z = agent.projector(z)
            z = F.normalize(z, dim=1, p=2)
            proto.append(z)
            sim = agent.protos(z)
            idx_ = sim.argmax()
            actual_proto.append(protos[idx_][None,:])
            
    print('data from buffer: therere {} unique prototypes that are neighbors to {} datapoints'.format(len(set(lst_proto)), a.shape[0]))

    encoded = torch.cat(encoded,axis=0)
    proto = torch.cat(proto,axis=0)
    actual_proto = torch.cat(actual_proto,axis=0) 
    # encoded_vdist = torch.norm(encoded_v[:,None, :] - encoded[None,:, :], dim=2, p=2)
    proto_vdist = torch.norm(proto_v[:,None,:] - proto[None,:, :], dim=2, p=2)
    actual_proto_vdist = torch.norm(actual_proto_v[:,None,:] - proto[None,:, :], dim=2, p=2)
    # all_dists_encode_v, _encode_v = torch.topk(encoded_vdist, 10, dim=1, largest=False)
    all_dists_proto_v, _proto_v = torch.topk(proto_vdist, 10, dim=1, largest=False)
    actual_all_dists_proto_v, _actual_proto_v = torch.topk(actual_proto_vdist, 10, dim=1, largest=False)

    with torch.no_grad():
        proto_v_sim = agent.protos(proto_v)
        actual_proto_v_sim = agent.protos(actual_proto_v)
    all_dists_proto_v_sim, _proto_v_sim = torch.topk(proto_v_sim, 10, dim=1, largest=True)
    actual_all_dists_proto_v_sim, _actual_proto_v_sim = torch.topk(actual_proto_v_sim, 10, dim=1, largest=True)

    dist_matrices = [_proto_v, _actual_proto_v, _proto_v_sim, _actual_proto_v_sim]
    names = [f"{model_name}_proto_w_vel.gif", f"{model_name}_actual_proto_w_vel.gif", f"{model_name}_sim_proto_w_vel.gif", f"{model_name}_sim_actual_proto_w_vel.gif"]
    for index_, dist_matrix in enumerate(dist_matrices):
        filenames=[]
        for ix, x in enumerate(test_states):
            txt=''
            df = pd.DataFrame()
            for i in range(a.shape[0]+1):
                if i!=a.shape[0]:
                    df.loc[i,'x'] = a[i,0].cpu().numpy()
                    df.loc[i,'y'] = a[i,1].cpu().numpy()
                    if i in dist_matrix[ix,:]:
                        df.loc[i, 'c'] = 'red'
                        z=dist_matrix[ix,(dist_matrix[ix,:] == i).nonzero(as_tuple=True)[0]]
                        txt += ' ['+str(np.round(state[z][0],2))+','+str(np.round(state[z][1],2))+'] '
                    else:
                        df.loc[i,'c'] = 'black'
                else:
                    df.loc[i,'x'] = x[0].item()
                    df.loc[i,'y'] = x[1].item()
                    df.loc[i,'c'] = 'blue'


            plt.clf()
            fig, ax = plt.subplots()
            ax=sns.scatterplot(x="x", y="y",
                      hue="c", 
                      data=df,legend=False)
            ax.set_title("\n".join(wrap(txt,75)))
            if index_==0:
                file1= f"./knn_output/10nn_proto_goals_w_vel{ix}_model{model_name}.png"
            elif index_==1:
                file1= f"./knn_output/10nn_actual_proto_goals_w_vel{ix}_model{model_name}.png"
            elif index_==2:
                file1= f"./knn_output/10nn_sim_proto_goals_w_vel{ix}_model{model_name}.png"
            elif index_==3:
                file1= f"./knn_output/10nn_sim_actual_proto_goals_w_vel{ix}_model{model_name}.png"
            plt.savefig(file1)
            filenames.append(file1)

        if len(filenames)>100:
            filenames=filenames[:100]

        with imageio.get_writer("./knn_output/"+names[index_], mode='I') as writer:
            print('saving gifs')
            for file in filenames:
                image = imageio.imread(file)
                writer.append_data(image)
                print('appending')

        gif = imageio.mimread('./knn_output/'+names[index_])

        imageio.mimsave('./knn_output/'+names[index_], gif, fps=.5)

    ################################################################


    #no velocity goals 


    # encoded_no_vdist = torch.norm(encoded_no_v[:,None, :] - encoded[None,:, :], dim=2, p=2)
    proto_no_vdist = torch.norm(proto_no_v[:,None,:] - proto[None,:, :], dim=2, p=2)
    actual_proto_no_vdist = torch.norm(actual_proto_no_v[:,None,:] - proto[None,:, :], dim=2, p=2)

    # all_dists_encode_no_v, _encode_no_v = torch.topk(encoded_no_vdist, 10, dim=1, largest=False)
    all_dists_proto_no_v, _proto_no_v = torch.topk(proto_no_vdist, 10, dim=1, largest=False)
    all_dists_actual_proto_no_v, _actual_proto_no_v = torch.topk(actual_proto_no_vdist, 10, dim=1, largest=False)

    with torch.no_grad():
        proto_no_v_sim = agent.protos(proto_no_v)
        actual_proto_no_v_sim = agent.protos(actual_proto_no_v)
    all_dists_proto_no_v_sim, _proto_no_v_sim = torch.topk(proto_no_v_sim, 10, dim=1, largest=True)
    actual_all_dists_proto_no_v_sim, _actual_proto_no_v_sim = torch.topk(actual_proto_no_v_sim, 10, dim=1, largest=True)


    dist_matrices = [_proto_no_v, _actual_proto_no_v, _proto_no_v_sim, _actual_proto_no_v_sim]
    names = [f"{model_name}_proto_no_vel.gif", f"{model_name}_actual_proto_no_vel.gif", f"{model_name}_sim_proto_no_vel.gif", f"{model_name}_sim_actual_proto_no_vel.gif"]

    for index_, dist_matrix in enumerate(dist_matrices):
        filenames=[]
        for ix, x in enumerate(goal_array):
            txt=''
            df = pd.DataFrame()
            for i in range(a.shape[0]+1):
                if i!=a.shape[0]:
                    df.loc[i,'x'] = a[i,0].cpu().numpy()
                    df.loc[i,'y'] = a[i,1].cpu().numpy()
                    if i in dist_matrix[ix,:]:
                        df.loc[i, 'c'] = 'red'
                        z=dist_matrix[ix,(dist_matrix[ix,:] == i).nonzero(as_tuple=True)[0]]
                        txt += ' ['+str(np.round(state[z][0],2))+','+str(np.round(state[z][1],2))+'] '
                    else:
                        df.loc[i,'c'] = 'black'
                else:
                    df.loc[i,'x'] = x[0].item()
                    df.loc[i,'y'] = x[1].item()
                    df.loc[i,'c'] = 'blue'


            plt.clf()
            fig, ax = plt.subplots()
            ax=sns.scatterplot(x="x", y="y",
                      hue="c", 
                      data=df,legend=False)
            ax.set_title("\n".join(wrap(txt,75)))
            if index_==0:
                file1= f"./knn_output/10nn_proto_goals_no_vel{ix}_model{model_name}.png"
            elif index_==1:
                file1= f"./knn_output/10nn_actual_proto_goals_no_vel{ix}_model{model_name}.png"
            elif index_==2:
                file1= f"./knn_output/10nn_sim_proto_goals_no_vel{ix}_model{model_name}.png"
            elif index_==3:
                file1= f"./knn_output/10nn_sim_actual_proto_goals_no_vel{ix}_model{model_name}.png"
            plt.savefig(file1)
            filenames.append(file1)

        if len(filenames)>100:
            filenames=filenames[:100]
        with imageio.get_writer('./knn_output/'+names[index_], mode='I') as writer:
            for file in filenames:
                image = imageio.imread(file)
                writer.append_data(image)

        gif = imageio.mimread('./knn_output/'+names[index_])

        imageio.mimsave('./knn_output/'+names[index_], gif, fps=.5)




    #swap goal & rand 1000 samples?
    # encoded_dist = torch.norm(encoded[:,None, :] - encoded[None,:, :], dim=2, p=2)
    proto_dist = torch.norm(protos[:,None,:] - proto[None,:, :], dim=2, p=2)
    all_dists_proto, _proto = torch.topk(proto_dist, 10, dim=1, largest=False)

    with torch.no_grad():
        proto_sim = agent.protos(proto).T
    all_dists_proto_sim, _proto_sim = torch.topk(proto_sim, 10, dim=1, largest=True)


    dist_matrices = [_proto, _proto_sim]
    names = [f"{model_name}_prototypes.gif", f"{model_name}_prototypes_sim.gif"]

    for index_, dist_matrix in enumerate(dist_matrices):
        filenames=[]
        for ix, x in enumerate(protos):
            txt=''
            df = pd.DataFrame()
            for i in range(a.shape[0]+1):
                if i!=a.shape[0]:
                    df.loc[i,'x'] = a[i,0].cpu().numpy()
                    df.loc[i,'y'] = a[i,1].cpu().numpy()
                    if i in dist_matrix[ix,:]:
                        df.loc[i, 'c'] = 'red'
                        z=dist_matrix[ix,(dist_matrix[ix,:] == i).nonzero(as_tuple=True)[0]]
                        txt += ' ['+str(np.round(state[z][0],2))+','+str(np.round(state[z][1],2))+'] '
                    else:
                        df.loc[i,'c'] = 'black'



            plt.clf()
            fig, ax = plt.subplots()
            ax=sns.scatterplot(x="x", y="y",
                      hue="c", 
                      data=df,legend=False)
            ax.set_title("\n".join(wrap(txt,75)))
            if index_==0:
                file1= f"./knn_output/10nn_actual_prototypes{ix}_model{model_name}.png"
            elif index_==1:
                file1= f"./knn_output/10nn_actual_prototypes_sim{ix}_model{model_name}.png"

            plt.savefig(file1)
            filenames.append(file1)

        if len(filenames)>100:
            filenames=filenames[:100]
        with imageio.get_writer('./knn_output/'+names[index_], mode='I') as writer:
            for file in filenames:
                image = imageio.imread(file)
                writer.append_data(image)

        gif = imageio.mimread('./knn_output/'+names[index_])

        imageio.mimsave('./knn_output/'+names[index_], gif, fps=.5)



# df = pd.DataFrame()
# for ix in range(_encode.shape[0]):
#     df.loc[ix, 'x'] = goal_array[ix][0].item()
#     df.loc[ix, 'y'] = goal_array[ix][1].item()
#     df.loc[ix, 'encoder'] = (_encode[ix]==torch.as_tensor(_state[ix],device='cuda')).sum().item()
#     df.loc[ix, 'proto'] = (_proto[ix]==torch.as_tensor(_state[ix],device='cuda')).sum().item()

# result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['encoder']
# result.fillna(0, inplace=True)
# plt.clf()
# fig, ax = plt.subplots()
# sns.heatmap(result, cmap="Blues_r", ax=ax).invert_yaxis()

# plt.savefig(f"./encoder_10nn.png")

# result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['proto']
# result.fillna(0, inplace=True)
# plt.clf()
# fig, ax = plt.subplots()
# sns.heatmap(result, cmap="Blues_r", ax=ax).invert_yaxis()

# plt.savefig(f"./proto_10nn.png")


    


# df1 = pd.DataFrame()
# df1['distance'] = dist_goal.reshape((1000,))
# df1['index'] = df1.index

# df1 = df1.sort_values(by='distance')

# goal_array_ = []
# for x in range(len(df1)):
#     goal_array_.append(goal_array[df1.iloc[x,1]])



# filenames=[]
# for ix, x in enumerate(goal_array):
#     txt = ''
#     df = pd.DataFrame()
#     count=0
#     for i in range(a.shape[0]+1):
#         if i!=a.shape[0]:
            
#             df.loc[i,'x'] = a[i,0].cpu().numpy()
#             df.loc[i,'y'] = a[i,1].cpu().numpy()

#             if i in _encode_no_v[ix,:]:
#                 count+=1
#                 df.loc[i, 'c'] = 'red'
                
#                 z=_encode_no_v[ix,(_encode_no_v[ix,:] == i).nonzero(as_tuple=True)[0]]
                
#                 txt += ' ['+str(np.round(state[z][0],2))+','+str(np.round(state[z][1],2))+'] '
#             else:
#                 df.loc[i,'c'] = 'black'

#         else:
#             df.loc[i,'x'] = x[0].item()
#             df.loc[i,'y'] = x[1].item()
#             df.loc[i,'c'] = 'blue'

#     if count<10:
#         import IPython as ipy; ipy.embed(colors='neutral')
#     plt.clf()
#     fig, ax = plt.subplots()
#     ax=sns.scatterplot(x="x", y="y",
#               hue="c",
#               data=df,legend=False)

#     print(txt)
#     ax.set_title("\n".join(wrap(txt,75)))
#     file1= f"./knn_output/10nn_goal_encode_no_vel{ix}.png"
#     plt.savefig(file1)
#     filenames.append(file1)


        
# final_encoded = np.array([i.detach().clone().cpu().numpy() for i in encoded])
# final_proto = np.array([i.detach().clone().cpu().numpy() for i in proto])
# final_encoded_goal = np.array([i.detach().clone().cpu().numpy() for i in encoded_goal])
# final_proto_goal = np.array([i.detach().clone().cpu().numpy() for i in proto_goal])   


# fn = [Path('./knn_visual/encoded_samples.npz'), Path('./knn_visual/proto_samples.npz'),
#       Path('./knn_visual/encoded_goals.npz'), Path('./knn_visual/proto_samples.npz')]
# lst_encoded = [final_encoded, final_proto, final_encoded_goal, final_proto_goal]

# for ix,x in enumerate(fn):
#     with io.BytesIO() as bs:
#         np.savez_compressed(bs, lst_encoded[ix])
#         bs.seek(0)
#         with x.open("wb") as f:
#             f.write(bs.read())
        


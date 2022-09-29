
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
encoder = torch.load('/home/nina/proto_explore/url_benchmark/model/encoder_proto_1000000.pth')
agent = torch.load('/home/nina/proto_explore/url_benchmark/model/optimizer_proto_1000000.pth')
#agent  = torch.load('/home/ubuntu/proto_explore/url_benchmark/exp_local/2022.09.09/072830_proto/optimizer_proto_1000000.pth')
#encoder  = torch.load('/home/ubuntu/proto_explore/url_benchmark/exp_local/2022.09.09/072830_proto/encoder_proto_1000000.pth')
eval_env_goal = dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None)




#replay_dir = Path('/home/ubuntu/proto_explore/url_benchmark/exp_local/2022.09.09/072830_proto/buffer2/buffer_copy/')
# replay_dir = Path('/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.09.21/150106_proto/buffer2/buffer_copy/')
replay_dir = Path('/home/nina/proto_explore/url_benchmark/model/buffer2/')
replay_buffer = make_replay_offline(eval_env_goal,
                                        replay_dir,
                                        1000,
                                        0,
                                        0,
                                        .99,
                                        goal=False,
                                        relabel=False,
                                        model_step = 2000000,
                                        replay_dir2=False,
                                        obs_type = 'pixels'
                                        )

state, actions, rewards = replay_buffer.parse_dataset()
idx = np.random.randint(0, state.shape[0], size=512)
state=state[idx]
state=state.reshape(512,4)
a = state
fn = Path('./knn_visual/samples.npz')
with io.BytesIO() as bs:
    np.savez_compressed(bs, a)
    bs.seek(0)
    with fn.open("wb") as f:
        f.write(bs.read())
        

def ndim_grid(ndims, space):
    L = [np.linspace(-.3,.3,space) for i in range(ndims)]
    return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T
lst=[]
goal_array = ndim_grid(2,10)
for ix,x in enumerate(goal_array):
    print(x[0])
    print(x[1])
    if (-.2<x[0]<.2 and -.02<x[1]<.02) or (-.02<x[0]<.02 and -.2<x[1]<.2):
        lst.append(ix)
        print('del',x)
goal_array=np.delete(goal_array, lst,0)
goal_array = ndim_grid(2,10)
emp = np.zeros((goal_array.shape[0],2))
goal_array = np.concatenate((goal_array, emp), axis=1)
print(goal_array.shape)
plt.clf()
fig, ax = plt.subplots()
ax.scatter(goal_array[:,0], goal_array[:,1])
plt.savefig(f"./mesh.png")
lst=[]
goal_array = torch.as_tensor(goal_array, device=torch.device('cuda'))
#a = np.random.uniform(-.29,.29, size=(1500,2))
#for ix,x in enumerate(a):
#    if (-.2<x[0]<.2 and -.02<x[1]<.02) or (-.02<x[0]<.02 and -.2<x[1]<.2):
#        lst.append(ix)
#        print('del',x)
#a=np.delete(a, lst,0)
#
print(a.shape)

plt.clf()
fig, ax = plt.subplots()
ax.scatter(a[:,0], a[:,1])
plt.savefig(f"./samples.png")
a = torch.as_tensor(a,device=torch.device('cuda'))

state_dist = torch.norm(goal_array[:,None,:]  - a[None,:,:], dim=2, p=2)
all_dists_state, _state = torch.topk(state_dist, 10, dim=1, largest=False)

encoded = []
proto = []
encoded_goal = []
proto_goal = []

for x in goal_array:
    with torch.no_grad():
        with eval_env_goal.physics.reset_context():
            time_step_init = eval_env_goal.physics.set_state(np.array([x[0].item(), x[1].item(),0,0]))
        
        time_step_init = eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
        time_step_init = np.transpose(time_step_init, (2,0,1))
        time_step_init = np.tile(time_step_init, (3,1,1))
        
        #print(eval_env_goal._env.physics.get_state())
        obs = time_step_init
        obs = torch.as_tensor(obs.copy(), device=torch.device('cuda')).unsqueeze(0)
        z = agent.encoder(obs)
        encoded_goal.append(z)
        z = agent.predictor(z)
        z = agent.projector(z)
        z = F.normalize(z, dim=1, p=2)
        proto_goal.append(z)
        
for x in a:

    with torch.no_grad():
        with eval_env_goal.physics.reset_context():
            time_step_init = eval_env_goal.physics.set_state(np.array([x[0].item(), x[1].item(),0,0]))
        
        time_step_init = eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
        time_step_init = np.transpose(time_step_init, (2,0,1))
        time_step_init = np.tile(time_step_init, (3,1,1))
        
        obs = time_step_init
        obs = torch.as_tensor(obs, device=torch.device('cuda')).unsqueeze(0)
        z = agent.encoder(obs)
        encoded.append(z)
        z = agent.predictor(z)
        z = agent.projector(z)
        z = F.normalize(z, dim=1, p=2)
        proto.append(z)
        
        
final_encoded = np.array([i.detach().clone().cpu().numpy() for i in encoded])
final_proto = np.array([i.detach().clone().cpu().numpy() for i in proto])
final_encoded_goal = np.array([i.detach().clone().cpu().numpy() for i in encoded_goal])
final_proto_goal = np.array([i.detach().clone().cpu().numpy() for i in proto_goal])   


fn = [Path('./knn_visual/encoded_samples.npz'), Path('./knn_visual/proto_samples.npz'),
      Path('./knn_visual/encoded_goals.npz'), Path('./knn_visual/proto_samples.npz')]
lst_encoded = [final_encoded, final_proto, final_encoded_goal, final_proto_goal]

for ix,x in enumerate(fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, lst_encoded[ix])
        bs.seek(0)
        with x.open("wb") as f:
            f.write(bs.read())
        
encoded = torch.cat(encoded,axis=0)
proto = torch.cat(proto,axis=0)
encoded_goal = torch.cat(encoded_goal,axis=0)
proto_goal = torch.cat(proto_goal,axis=0)

        #swap goal & rand 1000 samples?
encoded_to_goal = torch.norm(encoded_goal[:, None, :] - encoded[None, :, :], dim=2, p=2)
proto_to_goal = torch.norm(proto_goal[:, None, :] - proto[None, :, :], dim=2, p=2)
all_dists_encode, _encode = torch.topk(encoded_to_goal, 10, dim=1, largest=False)
all_dists_proto, _proto = torch.topk(proto_to_goal, 10, dim=1, largest=False)

filenames=[]
for ix, x in enumerate(goal_array):
    print('goal',ix)
    num=11
    colors = ['blue'] + ['red']*10 + ['black']*(a.shape[0]-11)
    data = np.empty((a.shape[0],4))
    data[0] = x.cpu().numpy()
    goal_array_copy = torch.cat([goal_array[0:ix], goal_array[ix+1:]])
    for iz in range(1,a.shape[1]-1):
        if iz <=10:
            data[iz] = a[_encode[ix,iz-1].item()].cpu().numpy()
    for iz in range(11,a.shape[0]):
#         print(iz)
        if iz-num in _encode[ix,:]:
            num -=1
            continue
        else:
            data[iz] = a[iz-num].cpu().numpy()
#             print(data[iz])

    
    for xy, color in zip(data, colors):
        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(xy[0], xy[1], 'o', color=color, picker=True)
# #     sns.set(style="ticks")
# #     data = sorted(zip(data, colors), key=lambda x: x[1])
# #     data_1 = [x[0] for x in data]
# #     data_2 = [x[1] for x in data]
#     xy = pd.DataFrame()
#     for ix, x in enumerate(data):
#         xy.loc[ix,'x'] = data[ix][1]
#         xy.loc[ix,'y'] = data[ix][2]
#         xy.loc[ix,'color'] = colors[ix]

    
#     sns.scatterplot(x=xy['x'], y=xy['y'], hue=xy['color'])
# #     sns.scatterplot(xy, data_2)
#         #ax.plot(xy[0], xy[1], 'o', color=color, picker=True)
    file1= f"./10nn_goal_encode{ix}.png"
    plt.savefig(file1)
    filenames.append(file1)

for file in filenames:
    image = imageio.imread(file)
    imageio.mimsave(os.path.join('./encoder_knn.gif'), image, duration = 0.9)

filenames=[]
for ix, x in enumerate(goal_array):
    print('goal proto',ix)
    num=11
    colors = ['blue'] + ['red']*10 + ['black']*(a.shape[0]-11)
    data = np.empty((a.shape[0],4))
    data[0] = x.cpu().numpy()
    for iz in range(1,a.shape[1]-1):
        if iz <=10:
            data[iz] = a[_proto[ix,iz-1].item()].cpu().numpy()
    for iz in range(11,a.shape[0]):
#         print(iz)
        if iz-num in _proto[ix,:]:
            num -=1
            continue
        else:
            data[iz] = a[iz-num].cpu().numpy()
#             print(data[iz])

    
    for xy, color in zip(data, colors):
        plt.clf()
        fig, ax = plt.subplots()
        ax.plot(xy[0], xy[1], 'o', color=color, picker=True)
# #     sns.set(style="ticks")
# #     data = sorted(zip(data, colors), key=lambda x: x[1])
# #     data_1 = [x[0] for x in data]
# #     data_2 = [x[1] for x in data]
#     xy = pd.DataFrame()
#     for ix, x in enumerate(data):
#         xy.loc[ix,'x'] = data[ix][1]
#         xy.loc[ix,'y'] = data[ix][2]
#         xy.loc[ix,'color'] = colors[ix]

    
#     sns.scatterplot(x=xy['x'], y=xy['y'], hue=xy['color'])
# #     sns.scatterplot(xy, data_2)
#         #ax.plot(xy[0], xy[1], 'o', color=color, picker=True)
    file1= f"./10nn_goal_proto{ix}.png"
    plt.savefig(file1)
    filenames.append(file1)
    
for file in filenames:
    image = imageio.imread(file)
    imageio.mimsave(os.path.join('./proto_knn.gif'), image, duration = 0.9)
    

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




import glob
import seaborn as sns
import pandas as pd
import re
import natsort
import random
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
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

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS

tmux_session = '0_lambda'

work_path = str(os.getcwd().split('/')[-2])+'/'+str(os.getcwd().split('/')[-1])
exp_name = '_'.join([
    'exp', 'proto_eval', 'pmm', 'pixels', str(tmux_session),work_path
])
wandb.init(project="urlb", group='proto_eval', name='exp')
#encoder = torch.load('/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/encoder/2022.09.09/072830_proto_lambda/encoder_proto_1000000.pth')
agent  = torch.load('/home/ubuntu/proto_explore/url_benchmark/exp_local/2022.09.09/072830_proto/optimizer_proto_1000000.pth')
eval_env_goal = dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None)
def ndim_grid(ndims, space):
    L = [np.linspace(-.3,.3,space) for i in range(ndims)]
    return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T

goal_array = torch.as_tensor(ndim_grid(2,10))
a = np.random.uniform(-.29,.29, size=(2000,2))
a = torch.as_tensor(a)
# b = np.zeros((1000,2))
# states = np.concatenate((a,b), axis=1)

# dist_goal = cdist(np.array([[-.15, .29]]), a, 'euclidean')
state_dist = torch.norm(goal_array[:,None,:]  - a[None,:,:], dim=2, p=2)
all_dists_state, _state = torch.topk(state_dist, 10, dim=1, largest=False)

encoded = []
proto = []
encoded_goal = []
proto_goal = []

for x in goal_array:
    with torch.no_grad():
        with eval_env_goal.physics.reset_context():
            time_step_init = eval_env_goal.physics.set_state(np.array([x]))
        
        time_step_init = eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
        time_step_init = np.transpose(time_step_init, (2,0,1))

        obs = time_step_init
        obs = torch.as_tensor(obs, device=torch.device(cuda)).unsqueeze(0)
        z = agent.encoder(obs)
        encoded_goal.append(z)
        z = agent.predictor(z)
        z = agent.projector(z)
        z = F.normalize(z, dim=1, p=2)
        proto_goal.append(z)
        
for x in a:
    with torch.no_grad():
        with eval_env_goal.physics.reset_context():
            time_step_init = eval_env_goal.physics.set_state(np.array([x]))
        
        time_step_init = eval_env_goal._env.physics.render(height=84, width=84, camera_id=dict(quadruped=2).get('point_mass_maze', 0))
        time_step_init = np.transpose(time_step_init, (2,0,1))
        
        obs = self.time_step1.observation['pixels']
        obs = torch.as_tensor(obs, device=torch.device(cuda)).unsqueeze(0)
        z = agent.encoder(obs)
        encoded.append(z)
        z = agent.predictor(z)
        z = agent.projector(z)
        z = F.normalize(z, dim=1, p=2)
        proto.append(z)

        #swap goal & rand 1000 samples?
encoded_to_goal = torch.norm(encoded_goal[:, None, :] - encoded[None, :, :], dim=2, p=2)
proto_to_goal = torch.norm(proto_goal[:, None, :] - proto[None, :, :], dim=2, p=2)
all_dists_encode, _encode = torch.topk(encoded_to_goal, 10, dim=1, largest=False)
all_dists_proto, _proto = torch.topk(proto_to_goal, 10, dim=1, largest=False)

df = pd.DataFrame(columns=['x','y','encoder', 'proto'])
for ix in range(_encode.shape[0]):
    df.loc[ix, 'x'] = goal_array[ix][0]
    df.loc[ix, 'y'] = goal_array[ix][1]
    df.loc[ix, 'encoder'] = (_encode[ix]==_state[ix]).sum()
    df.loc[ix, 'proto'] = (_proto[ix]==_state[ix]).sum()

result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['encoder']
result.fillna(0, inplace=True)
plt.clf()
fig, ax = plt.subplots()
sns.heatmap(result, cmap="Blues_r", ax=ax).invert_yaxis()

plt.savefig(f"./encoder_10nn.png")
wandb.save(f"./encoder_10nn.png")

result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['proto']
result.fillna(0, inplace=True)
plt.clf()
fig, ax = plt.subplots()
sns.heatmap(result, cmap="Blues_r", ax=ax).invert_yaxis()

plt.savefig(f"./proto_10nn.png")
wandb.save(f"./proto_10nn.png")


    


# df1 = pd.DataFrame()
# df1['distance'] = dist_goal.reshape((1000,))
# df1['index'] = df1.index

# df1 = df1.sort_values(by='distance')

# goal_array_ = []
# for x in range(len(df1)):
#     goal_array_.append(goal_array[df1.iloc[x,1]])

df.to_csv('/home/ubuntu/proto_explore/url_benchmark/knn_encoder_eval.csv', index=False)


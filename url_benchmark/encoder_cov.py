import glob
import seaborn as sns
import pandas as pd
import re
import natsort
import random
import scipy
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
#sample 
# replay_dir = Path('/home/ubuntu/proto_explore/url_benchmark/exp_local/2022.09.09/072830_proto/buffer2/buffer_copy/')
# replay_dir = Path('/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/exp_local/2022.09.21/150106_proto/buffer2/buffer_copy/')
eval_env_goal = dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None)
env = dmc.make('point_mass_maze_reach_no_goal', 'pixels', 3, 2, seed=None, goal=None)
#replay_dir = Path('/home/nina/proto_explore/url_benchmark/exp_local/2022.10.12/112203_pggg/buffer1/buffer_copy/')
replay_dir = Path('/home/nina/proto_explore/url_benchmark/exp_local/2022.10.05/153101_proto_goal_gc_grid/buffer1/buffer_copy/')
replay_buffer = make_replay_offline(eval_env_goal,
                                        replay_dir,
                                        1000000,
                                        0,
                                        0,
                                        .99,
                                        goal=False,
                                        relabel=False,
                                        model_step = 1000000,
                                        replay_dir2=False,
                                        obs_type = 'pixels'
                                        )


state, proto, actions, rewards, eps, index = replay_buffer.parse_dataset(proto_goal=True)
state = state.reshape((state.shape[0],4))
print(state.shape)
proto = proto.reshape((proto.shape[0],128))
state_t = np.empty((50000,4))
proto_t = np.empty((50000,128))

rand = np.random.choice(state.shape[0], size=(50000,), replace=False)

for ix,i in enumerate(rand):
    state_t[ix] = state[i]

    proto_t[ix] = proto[i][None,:]

covar = np.cov(proto_t.T)
print(covar.shape)
U, S, Vh = scipy.linalg.svd(covar)
print(S)
print('rank',np.linalg.matrix_rank(covar))

    
    
        
    

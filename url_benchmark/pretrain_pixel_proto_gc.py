import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)
import itertools
import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
os.environ['HYDRA_FULL_ERROR']='1'
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
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_replay_buffer, ndim_grid
import matplotlib.pyplot as plt
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS


def make_agent(obs_type, obs_spec, action_spec, goal_shape, num_expl_steps, goal, cfg, 
                hidden_dim, batch_size,update_gc, lr,gc_only,offline, load_protos, task, frame_stack, action_repeat=2, 
               replay_buffer_num_workers=4, discount=.99, reward_scores=False, 
               num_seed_frames=4000, task_no_goal='point_mass_maze_reach_no_goal', 
               work_dir=None,goal_queue_size=10, tmux_session=None, eval_every_frames=10000, seed=None,
               eval_after_step=990000, episode_length=100, reward_nn=True, hybrid_gc=False, hybrid_pct=0,num_protos=512,
               stddev_schedule=.2, stddev_clip=.3, reward_scores_dense=False, reward_euclid=False, episode_reset_length=500,
               use_closest_proto=True,pos_reward=True,neg_reward=False, batch_size_gc=1024):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    cfg.goal_shape = goal_shape
    cfg.goal = goal
    cfg.lr = lr
    cfg.hidden_dim =hidden_dim
    cfg.batch_size = batch_size
    cfg.update_gc =update_gc
    cfg.gc_only = gc_only
    cfg.offline = offline
    cfg.load_protos = load_protos
    cfg.task = task
    cfg.frame_stack = frame_stack
    cfg.action_repeat = action_repeat
    cfg.replay_buffer_num_workers = replay_buffer_num_workers
    cfg.discount = discount
    cfg.reward_scores = reward_scores
    cfg.num_seed_frames = num_seed_frames
    cfg.task_no_goal = task_no_goal
    cfg.work_dir = work_dir
    cfg.goal_queue_size = goal_queue_size
    cfg.tmux_session = tmux_session
    cfg.eval_every_frames = eval_every_frames
    cfg.seed = seed
    cfg.eval_after_step = eval_after_step
    cfg.episode_length = episode_length
    cfg.reward_nn = reward_nn
    cfg.hybrid_gc=hybrid_gc
    cfg.hybrid_pct=hybrid_pct
    cfg.num_protos=512
    cfg.stddev_schedule=stddev_schedule
    cfg.stddev_clip=stddev_clip
    cfg.reward_scores_dense=reward_scores_dense
    cfg.reward_euclid=reward_euclid
    cfg.episode_reset_length=episode_reset_length
    cfg.use_closest_proto=use_closest_proto
    cfg.pos_reward=pos_reward
    cfg.neg_reward=neg_reward
    cfg.batch_size_gc=batch_size_gc
    return hydra.utils.instantiate(cfg)

def get_state_embeddings(agent, states):
    with torch.no_grad():
        s = agent.encoder(states)
        s = agent.predictor(s)
        s = agent.projector(s)
        s = F.normalize(s, dim=1, p=2)
    return s

def heatmaps(self, env, model_step, replay_dir2, goal):
    replay_dir = self.work_dir / 'buffer1' / 'buffer_copy'
        
    replay_buffer = make_replay_buffer(env,
                                Path(replay_dir),
                                2000000,
                                1,
                                0,
                                self.cfg.discount,
                                goal=goal,
                                relabel=False,
                                model_step=model_step,
                                replay_dir2=replay_dir2)
    
    states, actions, rewards = replay_buffer.parse_dataset()
    #only adding states and rewards in replay_buffer
    tmp = np.hstack((states, rewards))
    df = pd.DataFrame(tmp, columns= ['x', 'y', 'pos', 'v','r'])
    heatmap, _, _ = np.histogram2d(df.iloc[:, 0], df.iloc[:, 1], bins=50, 
                                   range=np.array(([-.29, .29],[-.29, .29])))
    plt.clf()
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax).invert_yaxis()
    ax.set_title(model_step)
    plt.savefig(f"./{self._global_step}_heatmap.png")
    
    #percentage breakdown
    df=df*100
    heatmap, _, _ = np.histogram2d(df.iloc[:, 0], df.iloc[:, 1], bins=20, 
                                   range=np.array(([-29, 29],[-29, 29])))
    plt.clf()
    fig, ax = plt.subplots(figsize=(10,10))
    labels = np.round(heatmap.T/heatmap.sum()*100, 1)
    sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax, annot=labels).invert_yaxis()
    plt.savefig(f"./{self._global_step}_heatmap_pct.png")
 
    #rewards seen thus far 
    df = df.astype(int)
    result = df.groupby(['x', 'y'], as_index=True).max().unstack('x')['r']
    result.fillna(0, inplace=True)
    plt.clf()
    fig, ax = plt.subplots()
    sns.heatmap(result, cmap="Blues_r", ax=ax).invert_yaxis()
    plt.savefig(f"./{self._global_step}_reward.png")


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        #if cfg.use_wandb:
        #    exp_name = '_'.join([
        #        cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type,
        #        str(cfg.seed)
        #    ])
        #    wandb.init(project="urlb", group=cfg.agent.name, name=exp_name)

#        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

        self.train_env1 = dmc.make(cfg.task_no_goal, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)

    
#     update_gc, offline, gc_only, task, obs_type, frame_steck, work_dir,action_repeat, batch_size, replay_buffer_num_workers,
#               nstep, discount, reward_scores, reward_euclid, num_seed_frames, task_no_goal, , goal_queue_size
        
    # create agent
        #import IPython as ipy; ipy.embed(colors='neutral')
        self.agent = make_agent(cfg.obs_type,
                                self.train_env1.observation_spec(),
                                self.train_env1.action_spec(),
                                (9, 84, 84),
                                cfg.num_seed_frames // cfg.action_repeat,
                                True,
                                cfg.agent,
                                cfg.hidden_dim,
                                cfg.batch_size,
                                cfg.update_gc,
                                cfg.lr,
                                False,
                                cfg.offline,
                                cfg.load_proto,
                                cfg.task,
                                cfg.frame_stack,
                                cfg.action_repeat,
                                replay_buffer_num_workers = cfg.replay_buffer_num_workers,
                                discount=cfg.discount,
                                reward_scores = cfg.reward_scores,
                                num_seed_frames = cfg.num_seed_frames,
                                task_no_goal=cfg.task_no_goal,
                                work_dir = self.work_dir,
                                goal_queue_size=cfg.goal_queue_size,
                                tmux_session=cfg.tmux_session,
                                eval_every_frames=cfg.eval_every_frames,
                                seed=cfg.seed,
                                eval_after_step=cfg.eval_after_step,
                                episode_length=cfg.episode_length,
                                reward_nn=cfg.reward_nn,
                                hybrid_gc=cfg.hybrid_gc,
                                hybrid_pct=cfg.hybrid_pct,
                                num_protos=cfg.num_protos,
                                stddev_schedule=cfg.stddev_schedule,
                                stddev_clip=cfg.stddev_clip,
                                reward_scores_dense=cfg.reward_scores_dense,
                                reward_euclid=cfg.reward_euclid,
                                episode_reset_length=cfg.episode_reset_length,
                                use_closest_proto=cfg.use_closest_proto,
                                pos_reward=cfg.pos_reward,
                                neg_reward=cfg.neg_reward)

        if self.cfg.load_encoder:
            #encoder = torch.load('/home/ubuntu/proto_explore/url_benchmark/exp_local/2022.09.09/072830_proto/encoder_proto_1000000.pth')
            #encoder = torch.load('/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/encoder/2022.09.09/072830_proto_lambda/encoder_proto_1000000.pth') 
            encoder = torch.load('/home/nina/proto_explore/url_benchmark/model/encoder_proto_1000000.pth')
            self.agent.init_encoder_from(encoder)
        if self.cfg.load_proto:
            proto = torch.load('/home/nina/proto_explore/url_benchmark/model/optimizer_proto_1000000.pth')
            #proto  = torch.load('/home/ubuntu/proto_explore/url_benchmark/exp_local/2022.09.09/072830_proto/optimizer_proto_1000000.pth')
            #proto = torch.load('/misc/vlgscratch4/FergusGroup/mortensen/proto_explore/url_benchmark/encoder/2022.09.09/072830_proto_lambda/optimizer_proto_1000000.pth')
            self.agent.init_protos_from(proto)  

        self.video_recorder = VideoRecorder(
           self.work_dir if cfg.save_video else None,
           camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
           use_wandb=self.cfg.use_wandb)
      #  self.train_video_recorder = TrainVideoRecorder(
      #      self.work_dir if cfg.save_train_video else None,
      #      camera_id=0 if 'quadruped' not in self.cfg.domain else 2,
      #      use_wandb=self.cfg.use_wandb)

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
    def replay_iter2(self):
        if self._replay_iter2 is None:
            self._replay_iter2 = iter(self.replay_loader2)
        return self._replay_iter2
    
    

    
        
    def train(self):
        # predicates
        resample_goal_every = 500
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        #self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step):


            # try to evaluate
            #if eval_every_step(self.global_step) and self.global_step!=0:
            #    print('trying to evaluate')
            #    self.eval()              

            self.agent.roll_out(self.global_step, self.cfg.curriculum)
            
            self._global_step += 1


            if self._global_step%100000==0 and self._global_step>=500000:
                print('saving agent')
                path = os.path.join(self.work_dir, 'encoder_{}_{}.pth'.format(str(self.cfg.agent.name),self._global_step))
                torch.save(self.agent.encoder, path)
                path = os.path.join(self.work_dir, 'critic1_{}_{}.pth'.format(str(self.cfg.agent.name),self._global_step))
                torch.save(self.agent.critic, path)
                path = os.path.join(self.work_dir, 'actor1_{}_{}.pth'.format(str(self.cfg.agent.name),self._global_step))
                torch.save(self.agent.actor, path)


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
    from pretrain_pixel_proto_gc import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()

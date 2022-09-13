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
from logger import Logger, save
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_replay_buffer, ndim_grid
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS


def make_agent(obs_type, obs_spec, action_spec, goal_shape, num_expl_steps, goal, cfg, 
                hidden_dim, batch_size,update_gc, lr,gc_only,offline, load_protos, task, frame_stack, action_repeat=2, 
               replay_buffer_num_workers=4, discount=.99, reward_scores=False, 
               reward_euclid=True, num_seed_frames=4000, task_no_goal='point_mass_maze_reach_no_goal', work_dir=None,goal_queue_size=10):
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
    cfg.reward_euclid = reward_euclid
    cfg.num_seed_frames = num_seed_frames
    cfg.task_no_goal = task_no_goal
    cfg.work_dir = work_dir
    cfg.goal_queue_size = goal_queue_size
    return hydra.utils.instantiate(cfg)

def get_state_embeddings(agent, states):
    with torch.no_grad():
        s = agent.encoder(states)
        s = agent.predictor(s)
        s = agent.projector(s)
        s = F.normalize(s, dim=1, p=2)
    return s

def heatmaps(self, env, model_step, replay_dir2, goal):
    if goal:
        replay_dir = self.work_dir / 'buffer1' / 'buffer_copy'
    else:
        replay_dir = self.work_dir / 'buffer2' / 'buffer_copy'
        
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
        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type,
                str(cfg.seed)
            ])
            wandb.init(project="urlb", group=cfg.agent.name, name=exp_name)


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
                                True,
                                cfg.offline,
                                cfg.load_proto,
                                cfg.task,
                                cfg.frame_stack,
                                cfg.action_repeat,
                                replay_buffer_num_workers = cfg.replay_buffer_num_workers,
                                discount=cfg.discount,
                                reward_scores = cfg.reward_scores,
                                reward_euclid = cfg.reward_euclid,
                                num_seed_frames = cfg.num_seed_frames,
                                task_no_goal=cfg.task_no_goal,
                                work_dir = self.work_dir,
                                goal_queue_size=cfg.goal_queue_size)

        if self.cfg.load_encoder:
            encoder = torch.load('/home/ubuntu/proto_explore/url_benchmark/exp_local/2022.09.09/072830_proto/encoder_proto_1000000.pth')
            self.agent.init_encoder_from(encoder)
        if self.cfg.load_proto:
            proto  = torch.load('/home/ubuntu/proto_explore/url_benchmark/exp_local/2022.09.09/072830_proto/optimizer_proto_1000000.pth')
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
    
    

    
    def encoding_grid(self):
        if self.loaded == False:
            replay_dir = self.work_dir / 'buffer2' / 'buffer_copy'
            self.replay_buffer_intr = make_replay_buffer(self.eval_env,
                                        replay_dir,
                                        100000,
                                        self.cfg.batch_size,
                                        0,
                                        self.cfg.discount,
                                        goal=False,
                                        relabel=False,
                                        model_step = self.global_step,
                                        replay_dir2=False,
                                        obs_type = self.cfg.obs_type
                                        )
            self.loaded = True
            pix, states, actions = self.replay_buffer_intr._sample(eval_pixel=True)
            pix = pix.astype(np.float64)
            states = states.astype(np.float64)
            states = states.reshape(-1,4)
            grid = pix.reshape(9, 84, 84)
            grid = torch.tensor(grid).cuda().float()
            return grid, states
        else:
            pix, states, actions = self.replay_buffer_intr._sample(eval_pixel=True)
            pix = pix.astype(np.float64)
            states = states.astype(np.float64)
            states = states.reshape(-1,4)
            grid = pix.reshape(9, 84, 84)
            grid = torch.tensor(grid).cuda().float()
            return grid, states

    def sample_goal_uniform(self, eval=False):
        if self.loaded_uniform == False:
            goal_index = pd.read_csv('/home/ubuntu/proto_explore/url_benchmark/uniform_goal_pixel_index.csv')
            for ix in range(len(goal_index)):
                tmp = np.load('/home/ubuntu/url_benchmark/models/pixels_proto_ddpg_cross/buffer2/buffer_copy/'+goal_index.iloc[ix, 0])
                self.uniform_goal.append(np.array(tmp['observation'][int(goal_index.iloc[ix, -1])]))
                self.uniform_state.append(np.array(tmp['state'][int(goal_index.iloc[ix, -1])]))
            self.loaded_uniform = True
            self.count_uniform +=1
            print('loaded in uniform goals')
            return self.uniform_goal[self.count_uniform-1], self.uniform_state[self.count_uniform-1][:2]
        else:
            if self.count_uniform<400:
                self.count_uniform+=1
            else:
                self.count_uniform = 1
            return self.uniform_goal[self.count_uniform-1], self.uniform_state[self.count_uniform-1][:2]
    
    def sample_goal_pixel(self, eval=False):
        replay_dir = self.work_dir / "buffer2" / "buffer_copy"
    #    if len(self.unreachable_goal) > 0 and eval==False:
    #        a = [tuple(row) for row in self.unreachable_state]
    #        idx = np.unique(a, axis=0, return_index=True)[1]
    #        self.unreachable_state = self.unreachable_state[idx]
    #        self.unreachable_goal = self.unreachable_goal[idx]
    #        print('list of unreachables', self.unreachable_state)
    #        obs = self.unreachable_goal[0]
    #        state = self.unreachable_state[0]
    #        self.unreachable_state = np.delete(self.unreachable_state, 0, 0)
    #        self.unreachable_goal = np.delete(self.unreachable_goal, 0, 0)
    #        return obs, state

        if (self.global_step<100000 and self.global_step%20000==0 and eval==False) or (self.global_step %100000==0 and eval==False):
            self.replay_buffer_goal = make_replay_buffer(self.eval_env,
                                                        replay_dir,
                                                        50000,
                                                        self.cfg.batch_size,
                                                        0,
                                                        self.cfg.discount,
                                                        goal=False,
                                                        relabel=False,
                                                         replay_dir2 = False,
                                                        obs_type=self.cfg.obs_type,
                                                        model_step=self.global_step                                                                                                          )
            obs, state = self.replay_buffer_goal._sample_pixel_goal(self.global_step)
            return obs, state
        else:
            obs, state = self.replay_buffer_goal._sample_pixel_goal(self.global_step)
            return obs, state

    def eval(self):
        heatmaps(self, self.eval_env, self.global_step, False, True)
        heatmaps(self, self.eval_env, self.global_step, False, False)

        for i in range(400):
            step, episode, total_reward = 0, 0, 0
            goal_pix, goal_state = self.sample_goal_uniform(eval=True)
            self.eval_env = dmc.make(self.cfg.task, self.cfg.obs_type, self.cfg.frame_stack,
                    self.cfg.action_repeat, seed=None, goal=goal_state)
            eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
            meta = self.agent.init_meta()
            while eval_until_episode(episode):
                time_step = self.eval_env.reset()
                #self.video_recorder.init(self.eval_env, enabled=(episode == 0))
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        if self.cfg.goal:
                            action = self.agent.act(time_step.observation['pixels'],
                                                    goal_pix,
                                                    meta,
                                                    self._global_step,
                                                    eval_mode=True)
                        else:
                            action = self.agent.act(time_step.observation,
                                                    meta,
                                                    self._global_step,
                                                    eval_mode=True)
                    time_step = self.eval_env.step(action)
                 #   self.video_recorder.record(self.eval_env)
                    total_reward += time_step.reward
                    step += 1
       
                episode += 1
               # self.video_recorder.save(f'{self.global_frame}.mp4')
            
                if self.cfg.eval:
                    print('saving')
                    save(str(self.work_dir)+'/eval_{}.csv'.format(model.split('.')[-2].split('/')[-1]), [[x.cpu().detach().numpy(), total_reward, time_step.observation[:2], step]])

                else:
                    print('saving')
                    print(str(self.work_dir)+'/eval_{}.csv'.format(self._global_step))
                    save(str(self.work_dir)+'/eval_{}.csv'.format(self._global_step), [[goal_state, total_reward, time_step.observation['observations'], step]])
        
#             if total_reward < 200*self.cfg.num_eval_episodes:
#                 self.unreachable_goal = np.append(self.unreachable_goal, np.array(goal_pix[None,:,:,:]), axis=0)
#                 self.unreachable_state = np.append(self.unreachable_state, np.array(goal_state[None,:]), axis=0)
                
    def eval_intrinsic(self, model):
        obs = torch.empty(1024, 9, 84, 84)
        states = torch.empty(1024, 4)
        grid_embeddings = torch.empty(1024, 128)
        actions = torch.empty(1024,2)
        meta = self.agent.init_meta()
        for i in range(1024):
            with torch.no_grad():
                grid, state = self.encoding_grid()
                action = self.agent.act2(grid, meta, self._global_step, eval_mode=True)
                actions[i] = action
                obs[i] = grid
                states[i] = torch.tensor(state).cuda().float()
        import IPython as ipy; ipy.embed(colors='neutral')    
        obs = obs.cuda().float()
        actions = actions.cuda().float()
        grid_embeddings = get_state_embeddings(self.agent, obs)
        protos = self.agent.protos.weight.data.detach().clone()
        protos = F.normalize(protos, dim=1, p=2)
        dist_mat = torch.cdist(protos, grid_embeddings)
        closest_points = dist_mat.argmin(-1)
        proto2d = states[closest_points.cpu(), :2]
        with torch.no_grad():
            reward = self.agent.compute_intr_reward(obs, self._global_step)
            q_value = self.agent.get_q_value(obs, actions)
        for x in range(len(reward)):
            print('saving')
            print(str(self.work_dir)+'/eval_intr_reward_{}.csv'.format(self._global_step))
            save(str(self.work_dir)+'/eval_intr_reward_{}.csv'.format(self._global_step), [[obs[x].cpu().detach().numpy(), reward[x].cpu().detach().numpy(), q[x].cpu().detach().numpy(), self._global_step]])

        
    def train(self):
        # predicates
        resample_goal_every = 500
        train_until_step = utils.Until(self.cfg.num_train_frames,
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

            self.agent.roll_out(self.global_step)
            
            self._global_step += 1

            #if self._global_step%50000==0 and self._global_step!=0:
            #    print('saving agent')
            #    path = os.path.join(self.work_dir, 'optimizer_{}_{}.pth'.format(str(self.cfg.agent.name),self._global_step))
            #    torch.save(self.agent, path)


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

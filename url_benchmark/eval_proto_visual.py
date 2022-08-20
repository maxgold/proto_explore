import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'
import glob
from pathlib import Path
import re
import hydra
import numpy as np
import torch
import wandb
from dm_env import specs
import matplotlib.pyplot as plt
from kdtree import KNN

import dmc
import torch.nn.functional as F
import utils
from logger import Logger, save
from replay_buffer import ReplayBufferStorage, make_replay_loader, make_replay_buffer
from video import TrainVideoRecorder, VideoRecorder
from agent.expert import ExpertAgent
import glob
torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS

# TODO:
# write code to sample from our generative model
# write code to import the expert gaol-conditioned agent
# write code to visualize prototyeps
#   to do this i think need to embed a grid of states
#   and then for each prototype find the closest state embedding to it

def ndim_grid(ndims, space):
    L = [np.linspace(-.25,.25,space) for i in range(ndims)]
    return np.hstack((np.meshgrid(*L))).swapaxes(0,1).reshape(ndims,-1).T

def get_state_embeddings(agent, states):
    with torch.no_grad():
        s = agent.encoder(states)
        s = agent.predictor(s)
        s = agent.projector(s)
        s = F.normalize(s, dim=1, p=2)
    return s


def visualize_prototypes(agent):
    grid = np.meshgrid(np.arange(-.3,.3,.01),np.arange(-.3,.3,.01))
    grid = np.concatenate((grid[0][:,:,None],grid[1][:,:,None]), -1)
    grid = grid.reshape(-1, 2)
    grid = np.c_[grid, np.zeros_like(grid)]
    grid = torch.tensor(grid).cuda().float()
    grid_embeddings = get_state_embeddings(agent, grid)
    protos = agent.protos.weight.data.detach().clone()
    protos = F.normalize(protos, dim=1, p=2)
    dist_mat = torch.cdist(protos, grid_embeddings)
    closest_points = dist_mat.argmin(-1)
    return grid[closest_points, :2].cpu()

def visualize_prototypes_visited(agent, replay_dir, cfg, env, model_step, replay_dir2):
    replay_buffer = make_replay_buffer(env,
                                    Path(cfg.replay_dir),
                                    cfg.replay_buffer_size,
                                    cfg.batch_size,
                                    0,
                                    cfg.discount,
                                    goal=True,
                                    relabel=False,
                                    model_step=model_step,
                                    replay_dir2=replay_dir2)
    states, actions = replay_buffer.parse_dataset()
    if states == '':
        print('nothing in buffer yet')
    else:
        states = states.astype(np.float64)
        grid = states.reshape(-1,4)
        grid = torch.tensor(grid).cuda().float()
        grid_embeddings = get_state_embeddings(agent, grid)
        protos = agent.protos.weight.data.detach().clone()
        protos = F.normalize(protos, dim=1, p=2)
        dist_mat = torch.cdist(protos, grid_embeddings)
        closest_points = dist_mat.argmin(-1)
        #import IPython as ipy; ipy.embed(colors='neutral')
        return grid[closest_points, :2].cpu()


def make_agent(obs_type, obs_spec, action_spec, goal_shape,num_expl_steps, goal, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    cfg.goal_shape = goal_shape
    cfg.goal = goal
    return hydra.utils.instantiate(cfg)

def make_generator(env, cfg):
    replay_dir = Path(
        "/home/maxgold/workspace/explore/proto_explore/url_benchmark/exp_local/2022.07.23/101256_proto/buffer2"
    )
    replay_buffer = make_replay_buffer(
        env,
        replay_dir,
        cfg.replay_buffer_size,
        cfg.batch_size,
        0,
        cfg.discount,
        goal=True,
        relabel=False,
    )
    states, actions, futures = replay_buffer.parse_dataset()
    states = states.astype(np.float64)
    knn = KNN(states[:, :2], futures)
    return knn


def make_expert():
    return ExpertAgent()


class Workspace:
    def __init__(self, cfg, agent):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        self.agent_path = agent
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        if cfg.use_wandb:
            exp_name = '_'.join([
                cfg.experiment, cfg.agent.name, cfg.domain, cfg.obs_type,
                str(cfg.seed)
            ])
            wandb.init(project="urlb", group=cfg.agent.name, name=exp_name)

        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        # create envs
        try:
            task = PRIMAL_TASKS[self.cfg.domain]
        except:
            task = self.cfg.domain
        self.train_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                  cfg.action_repeat, cfg.seed)
        self.eval_env = dmc.make(task, cfg.obs_type, cfg.frame_stack,
                                 cfg.action_repeat, cfg.seed)

        # create agent
        self.agent = torch.load(self.agent_path)

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))


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
        self.expert = make_expert()
        #self.knn = make_generator(self.eval_env, cfg)
        self.use_expert = False

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self._global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def sample_goal(self, obs):
        cands = self.knn.query_k(np.array(obs[:2])[None], 10)
        cands = torch.tensor(cands[0, :, :, 1]).cuda()
        with torch.no_grad():
            z = self.agent.encoder(cands)
            z = self.agent.predictor(z)
            z = F.normalize(z, dim=1, p=2)
            # this score is P x B and measures how close 
            # each prototype is to the elements in the batch
            # each prototype is assigned a sampled vector from the batch
            # and this sampled vector is added to the queue
            scores = self.agent.protos(z).T
            current_protos = self.agent.protos.weight.data.clone()
        current_protos = F.normalize(current_protos, dim=1, p=2)
        z_to_c = torch.norm(z[:, None, :] - current_protos[None, :, :], dim=2, p=2)
        all_dists, _ = torch.topk(z_to_c, 3, dim=1, largest=True)
        ind = all_dists.mean(-1).argmax().item()
        return cands[ind].cpu().numpy()

    
    def sample_goal_proto(self, obs):
        #current_protos = self.agent.protos.weight.data.clone()
        #current_protos = F.normalize(current_protos, dim=1, p=2)
        proto2d = visualize_prototypes(self.agent)
        num = proto2d.shape[0]
        idx = np.random.randint(0, num)
        proto2d = proto2d[idx, :]
        #import IPython as ipy; ipy.embed(colors='neutral')
        return proto2d[idx,:].cpu().numpy()
    def eval(self, path, model_step, replay_dir2):
        proto2d = visualize_prototypes_visited(self.agent, path, self.cfg, self.train_env, model_step, replay_dir2)
        plt.clf()
        fig, ax = plt.subplots()
        ax.scatter(proto2d[:,0], proto2d[:,1])
        plt.savefig(f"./{model_step}_proto2d.png")
    

    def eval_goal_proto(self, path, model_step, replay_dir2):
        #if cfg.eval, then eval over goals
        #final evaluation over all final prototypes
        #load final agent model to get them
        if self.cfg.eval:
            proto2d = visualize_prototypes(self.agent)
            num = proto2d.shape[0]
            print('proto2d', proto2d.shape)
            idx = np.random.randint(0, num,size=(50,))
            proto2d = proto2d[idx, :]
            plt.clf()
            fig, ax = plt.subplots()
            ax.scatter(proto2d[:,0], proto2d[:,1])
            plt.savefig(f"./final_proto2d.png")
        else:
            proto2d = visualize_prototypes_visited(self.agent, path, self.cfg, self.train_env, model_step, replay_dir2)
            #current prototypes of the training agent
            print('proto2d', proto2d.shape)
            num = proto2d.shape[0]
            #idx = np.random.randint(0, num,size=(50,))
            #proto2d = proto2d[idx, :]
            plt.clf()
            fig, ax = plt.subplots()
            ax.scatter(proto2d[:,0], proto2d[:,1])
            plt.savefig(f"./{self._global_step}_proto2d.png")
        
        for ix, x in enumerate(proto2d):
            print('goal', x)   
            print(ix)
            step, episode, total_reward = 0, 0, 0
            self.eval_env = dmc.make(self.cfg.task, seed=None, goal=x.cpu().detach().numpy())
            eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
            meta = self.agent.init_meta()
            
            while eval_until_episode(episode):
                
                time_step = self.eval_env.reset()
                #self.video_recorder.init(self.eval_env, enabled=(episode == 0))
                #goal = np.random.sample((2,)) * .5 - .25
                
                while not time_step.last():
                    
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        
                        if self.cfg.goal:
                            action = self.agent.act(time_step.observation,
                                                x,
                                                meta,
                                                self._global_step,
                                                eval_mode=True)
                        else:
                            action = self.agent.act(time_step.observation,
                                                meta,
                                                self._global_step,
                                                eval_mode=True)
                   
                    time_step = self.eval_env.step(action)
                    #self.video_recorder.record(self.eval_env)
                    total_reward += time_step.reward
                    step += 1

                #self.video_recorder.save(f'{self.global_frame}.mp4')

                episode += 1
            
                if self.cfg.eval:
                    print('saving')
                    save(str(self.work_dir)+'/eval_goal_proto_{}.csv'.format(model_step), [[x.cpu().detach().numpy(), total_reward, time_step.observation[:2], step]])
            
                else:
            
                    print('saving')
                    print(str(self.work_dir)+'/eval_{}.csv'.format(self._global_step))
                    save(str(self.work_dir)+'/eval_{}.csv'.format(self._global_step), [[x.cpu().detach().numpy(), total_reward, time_step.observation[:2], step]])
                




    def eval_goal(self, path, model_step, replay_dir2):
        goal_array = ndim_grid(2, 40)
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        for ix, x in enumerate(goal_array):
            step, episode, total_reward = 0, 0, 0
            env = dmc.make(self.cfg.task, seed=None, goal=x)
            time_step = env.reset()
            meta = self.agent.init_meta()
            while eval_until_episode(episode):
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        if self.cfg.goal:
                            action = self.agent.act(time_step.observation, x, meta, self.global_step, eval_mode=True)
                        else:
                            action = self.agent.act(time_step.observation, meta,self.global_step, eval_mode=True)

                    time_step = env.step(action)
                    total_reward += time_step.reward
                    step += 1

                episode += 1
                print('work_dir+...', str(self.work_dir))
                print('model', model_step)
                if self.cfg.eval:
                    print('saving')
                    save(str(self.work_dir)+'/eval_goal_{}.csv'.format(model_step), [[x, total_reward, time_step.observation[:2], step]])

                else:
                    print('saving')
                    print(str(self.work_dir)+'/eval_{}_{}.csv'.format(global_index, global_step))
                    save(str(self.work_dir)+'/eval_{}_{}.csv'.format(global_index, global_step), [[x, total_reward, time_step.observation[:2], step]])
        
    def eval_intr_reward(self, path, model_step, replay_dir2):
        goal_array = ndim_grid(4, 20)
        for i in range(1000):
            eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
            idx = np.random.randint(0, len(goal_array),size=(1024,))
            obs = goal_array[idx]
            reward = self.agent.compute_intr_reward(obs, model_step)
        
            if self.cfg.eval:
                print('saving')
                save(str(self.work_dir)+'/eval_intr_reward_{}.csv'.format(model_step), [[x, reward, x, model_step]])


    def train(self):
        # predicates
        resample_goal_every = 50
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        if self.global_step <100:
            goal = np.random.sample((2,)) * .5 - .25
            #figure out how to train two policies simultaneously, one maximizing intrinsic reward one maximizing extrinsic (goal conditioned)
        else:
            goal = self.sample_goal_proto(time_step.observation)[:2]
        self.train_env = dmc.make(self.cfg.task, seed=None, goal=goal)
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta, goal)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self._global_step):
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
                        log('step', self._global_step)

                # reset env
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                self.replay_storage.add_goal(time_step, meta, goal)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                    episode_step = 0
                    episode_reward = 0

            # try to evaluate
            if eval_every_step(self._global_step):
                if self.cfg.eval:
                    
                    model_lst = glob.glob(str(self.cfg.path)+'/*400000.pth')
                    if len(model_lst)>0:
                        print(model_lst[ix])
                        proto = torch.load(model_lst[ix])
                        self.eval_goal(proto, model_lst[ix])
                    
                    self.global_step = 500000
                    self.global_step +=1
                        
                else:
                    if self.global_step%100000==0 and self.global_step!=0:
                        proto=self.agent
                        model = ''
                        self.eval_goal(proto, model)
                    else:
                        self.logger.log('eval_total_time', self.timer.total_time(),
                                    self.global_frame)
                        self.eval()

            meta = self.agent.update_meta(meta, self._global_step, time_step)
            if episode_step % resample_goal_every == 0:
                goal = self.sample_goal_proto(time_step.observation)[:2]
                self.train_env = dmc.make(self.cfg.task, seed=None, goal=goal)
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                if self.cfg.goal:
                    action = self.agent.act(time_step.observation,
                                            goal,
                                            meta,
                                            self._global_step,
                                            eval_mode=False)
                else:
                    action = self.agent.act(time_step.observation,
                                            meta,
                                            self._global_step,
                                            eval_mode=False)

            # try to update the agent
            if not seed_until_step(self._global_step):
                metrics = self.agent.update(self.replay_iter, self._global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward
            self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1
            
            #save agent
            if self._global_step%100000==0:
                path = os.path.join(self.work_dir, 'optimizer_{}_{}.pth'.format(str(self.cfg.agent.name), self._global_step))
                torch.save(self.agent, path)
    
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
    from eval_proto_visual import Workspace as W
    root_dir = Path.cwd()
    agents = glob.glob(str(cfg.path)+'/*pth')
    print(agents)
    
    for ix, x in enumerate(agents):
        if int(re.findall('\d+',x)[-1])==1900000:
            workspace = W(cfg, x)
            model = str(x).split('_')[-1]
            model = str(model).split('.')[-2]
            replay_dir = Path(cfg.replay_dir)
            if cfg.replay_dir2:
                replay_dir2 = Path(cfg.replay_dir2)
            else:
                replay_dir2 = False
            print('model_step', model)
            #workspace.eval(replay_dir, model, replay_dir2)
            workspace.eval_goal(replay_dir, model, replay_dir2)
            #workspace.eval_intr_reward(replay_dir, model, replay_dir2)
            print(ix)

if __name__ == '__main__':
    main()
import warnings


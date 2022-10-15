import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

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
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from offline_replay_buffer import make_replay_buffer
from video import TrainVideoRecorder, VideoRecorder
from agent.expert import ExpertAgent

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS

# TODO:
# replace expert model with goal based model
# intermittently train goal based model
# modify replay buffer so that we can add trajectories to it and sample from it
# as well


def get_state_embeddings(agent, states):
    with torch.no_grad():
        s = agent.encoder(states)
        s = agent.predictor(s)
        s = agent.projector(s)
        s = F.normalize(s, dim=1, p=2)
    return s


def visualize_prototypes(agent, states=None):
    if states is None:
        grid = np.meshgrid(np.arange(-0.3, 0.3, 0.01), np.arange(-0.3, 0.3, 0.01))
        grid = np.concatenate((grid[0][:, :, None], grid[1][:, :, None]), -1)
        grid = grid.reshape(-1, 2)
        grid = np.c_[grid, np.zeros_like(grid)]
        grid = torch.tensor(grid).cuda().float()
    else:
        grid = np.array(states)
        grid = torch.tensor(grid).to(agent.device)
    grid_embeddings = get_state_embeddings(agent, grid)
    protos = agent.protos.weight.data.clone()
    protos = F.normalize(protos, dim=1, p=2)
    dist_mat = torch.cdist(protos, grid_embeddings)
    closest_points = dist_mat.argmin(-1)
    return grid[closest_points, :2].cpu()


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg, goal_shape=(2,))


def make_goal_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    from agent.gcsl import GCSLAgent

    goal_agent = GCSLAgent(
        "gcsl",
        cfg.obs_shape,
        cfg.action_shape,
        (2,),
        cfg.device,
        cfg.lr,
        cfg.hidden_dim,
        cfg.stddev_schedule,
        cfg.nstep,
        cfg.batch_size,
        cfg.stddev_clip,
        cfg.use_tb,
        False,
    )
    return goal_agent


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
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        if cfg.use_wandb:
            exp_name = "_".join(
                [
                    cfg.experiment,
                    cfg.agent.name,
                    cfg.domain,
                    cfg.obs_type,
                    str(cfg.seed),
                ]
            )
            wandb.init(project="proto_explore", group=cfg.tag, name=exp_name)

        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)
        # create envs
        try:
            task = PRIMAL_TASKS[self.cfg.domain]
        except:
            task = self.cfg.domain
        self.train_env = dmc.make(
            task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed
        )
        self.eval_env = dmc.make(
            task, cfg.obs_type, cfg.frame_stack, cfg.action_repeat, cfg.seed
        )

        # create agent
        self.agent = make_agent(
            cfg.obs_type,
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            cfg.num_seed_frames // cfg.action_repeat,
            cfg.agent,
        )
        self.goal_agent = make_goal_agent(
            cfg.obs_type,
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            cfg.num_seed_frames // cfg.action_repeat,
            cfg.agent,
        )

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
        )

        # create data storage
        self.replay_storage = ReplayBufferStorage(
            data_specs, meta_specs, self.work_dir / "buffer"
        )

        # create replay buffer
        self.replay_loader = make_replay_loader(
            self.replay_storage,
            cfg.replay_buffer_size,
            cfg.batch_size,
            cfg.replay_buffer_num_workers,
            False,
            cfg.nstep,
            cfg.discount,
            goal=True,
        )

        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0 if "quadruped" not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb,
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0 if "quadruped" not in self.cfg.domain else 2,
            use_wandb=self.cfg.use_wandb,
        )

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self.expert = make_expert()
        self.knn = make_generator(self.eval_env, cfg)
        self.use_expert = self.cfg.use_expert

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
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def sample_goal(self, obs):
        if False:
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
        goal = np.random.rand(2) * 0.6 - 0.3
        #goal = np.random.rand(2) * 0.3
        goal[1] = -goal[1]
        return goal

    def eval(self, states):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(
                        time_step.observation, meta, self.global_step, eval_mode=True
                    )
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f"{self.global_frame}.mp4")
        # if self.global_step % int(1e5) == 0:
        if len(states):
            proto2d = visualize_prototypes(self.agent, states)
            plt.clf()
            fig, ax = plt.subplots()
            ax.scatter(proto2d[:, 0], proto2d[:, 1])
            plt.savefig(f"./{self.global_step}_proto2d.png")
            data = [[x, y] for (x, y) in zip(proto2d[:, 0], proto2d[:, 1])]
            table = wandb.Table(data=data, columns=["x", "y"])
            wandb.log(
                {
                    f"viz/proto{int(self.global_step/self.cfg.eval_every_frames)}": wandb.plot.scatter(
                        table,
                        "x",
                        "y",
                        title=f"Prototypes {int(self.global_step/self.cfg.eval_every_frames)}",
                    )
                }
            )

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("episode_length", step * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

    def train(self):
        # predicates
        resample_goal_every = 200
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        if self.cfg.goal:
            goal = self.sample_goal(time_step.observation)[:2]
        meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        states = []
        while train_until_step(self.global_step):
            if time_step.last():
                self._global_episode += 1
                self.train_video_recorder.save(f"{self.global_frame}.mp4")
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("buffer_size", len(self.replay_storage))
                        log("step", self.global_step)

                # reset env
                time_step = self.train_env.reset()
                meta = self.agent.init_meta()
                self.replay_storage.add(time_step, meta)
                self.train_video_recorder.init(time_step.observation)
                # try to save snapshot
                if self.global_frame in self.cfg.snapshots:
                    self.save_snapshot()
                episode_step = 0
                episode_reward = 0

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval(states)

            meta = self.agent.update_meta(meta, self.global_step, time_step)
            if episode_step % resample_goal_every == 0:
                if self.cfg.goal:
                    goal = self.sample_goal(time_step.observation)[:2]
            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                if self.use_expert:
                    action = self.expert.act(
                        time_step.observation, goal, self.global_step, eval_mode=False
                    )
                else:
                    action = self.agent.act(
                        time_step.observation,
                        meta,
                        self.global_step,
                        eval_mode=False,
                    )

            # try to update the agent
            if not seed_until_step(self.global_step):
                if self.cfg.goal:
                    batch = next(self.replay_iter)
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                    #metrics = self.agent.update2(batch, self.global_step)
                    metrics.update(self.goal_agent.update(batch, self.global_step))
                else:
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # take env step
            time_step = self.train_env.step(action)
            states.append(time_step.observation)
            episode_reward += time_step.reward
            # time_step.observation = time_step.observation.astype(np.float32)
            self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f"snapshot_{self.global_frame}.pt"
        keys_to_save = ["agent", "goal_agent", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)


@hydra.main(config_path=".", config_name="pretrain_goal")
def main(cfg):
    from pretrain_goal import Workspace as W

    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    print("training")
    workspace.train()


if __name__ == "__main__":
    main()

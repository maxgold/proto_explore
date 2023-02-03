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
import torch.nn.functional as F
import matplotlib.pyplot as plt

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
import pickle
import tqdm
import math


torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS
from agent.gcsl import GCSLAgent2
from agent.expert import ExpertAgent


def get_encoding(agent, time_step, num_aug=5):
    obs = torch.tensor(time_step.observation).cuda()[None]
    auglist = []
    with torch.no_grad():
        for _ in range(num_aug):
            auglist.append(agent.aug_and_encode(obs))
    return torch.mean(torch.stack(auglist), 0).cpu()


def make_agent(obs_type, obs_spec, action_spec, num_expl_steps, cfg):
    cfg.obs_type = obs_type
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    cfg.num_expl_steps = num_expl_steps
    return hydra.utils.instantiate(cfg)


def sample_sphere(num_points, width=1):
    samples = np.random.rand(num_points) * 2 * np.pi
    pts = np.c_[np.cos(samples)[:,None], np.sin(samples)[:,None]]
    return pts * width


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)

        # create logger
        self.logger = Logger(self.work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)
        # create envs

        if "cheetah" in cfg.task:
            self.train_env = dmc.make(
                cfg.task,
                cfg.obs_type,
                cfg.frame_stack,
                cfg.action_repeat,
                cfg.seed,
                time_limit=10,
            )
            self.eval_env = dmc.make(
                cfg.task,
                cfg.obs_type,
                cfg.frame_stack,
                cfg.action_repeat,
                cfg.seed,
                time_limit=10,
            )
        elif "walker" in cfg.task:
            self.train_env = dmc.make(
                cfg.task,
                cfg.obs_type,
                cfg.frame_stack,
                cfg.action_repeat,
                cfg.seed,
                time_limit=25,
            )
            self.eval_env = dmc.make(
                cfg.task,
                cfg.obs_type,
                cfg.frame_stack,
                cfg.action_repeat,
                cfg.seed,
                time_limit=25,
            )
        else:
            self.train_env = dmc.make(
                cfg.task,
                cfg.obs_type,
                cfg.frame_stack,
                cfg.action_repeat,
                cfg.seed,
                time_limit=20,
            )
            self.eval_env = dmc.make(
                cfg.task,
                cfg.obs_type,
                cfg.frame_stack,
                cfg.action_repeat,
                cfg.seed,
                time_limit=20,
            )

        # create agent
        self.agent = make_agent(
            cfg.obs_type,
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            cfg.num_seed_frames // cfg.action_repeat,
            cfg.agent,
        )

        # initialize from pretrained
        if cfg.snapshot_ts > 0:
            pretrained_agent = self.load_snapshot()["agent"]
            self.agent.init_from(pretrained_agent)

        #        with open("/home/maxgold/workspace/explore/proto_explore/url_benchmark/play_dataset.pkl", "rb") as f:
        #            self.play_dataset = pickle.load(f)

        self.agent.encoder.cuda()
        #        num_aug = 5
        #        encodings = []
        #        actions = []
        #        physics = []
        #
        #        for tmp2 in tqdm.tqdm(self.play_dataset):
        #            tmp = tmp2["obs"]
        #            actions.append(torch.tensor(tmp2["action"]))
        #            physics.append(torch.tensor(tmp2["physics"]))
        #            tmp = torch.stack([torch.tensor(o).cuda() for o in tmp])
        #            auglist = []
        #            with torch.no_grad():
        #                for _ in range(num_aug):
        #                    auglist.append(self.agent.aug_and_encode(tmp))
        #            encodings.append(torch.mean(torch.stack(auglist), 0).cpu())

        # self.encodings = torch.stack(encodings)
        # self.actions = torch.stack(actions)
        # self.physics = torch.stack(physics)
        self.buffer_size = 1000
        self.encodings = torch.zeros(self.buffer_size, 500, self.agent.encoder.repr_dim)
        self.actions = torch.zeros(self.buffer_size, 499, 2)
        self.physics = torch.zeros(self.buffer_size, 500, 4)

        oshape = (self.encodings.shape[2],)
        ashape = (self.actions.shape[2],)
        oshape = (4,)
        ashape = (2,)
        lr = 1e-4

        self.goal_agent = GCSLAgent2(oshape, ashape, oshape, "cuda", lr, 256)
        self.expert = ExpertAgent()
        self.cid = 0

        # TODO:
        # [X] parse play_dataset (encode with encoder)
        # [X] figure out dimensions to instantiate GCSLAgent2 with
        # [X] sample goals less than 10 timesteps apart
        # train!

    def sample(self, num, max_horizon=20):
        xinds = np.random.choice(
            range(min(self.encodings.shape[0], self.cid)), size=num, replace=True
        )
        startinds = np.random.choice(
            range(self.encodings.shape[1] - max_horizon), size=num, replace=True
        )
        endinds = startinds + np.random.choice(
            range(max_horizon), size=num, replace=True
        )

        #o1 = self.encodings[xinds, startinds]
        #o2 = self.encodings[xinds, endinds]
        o1 = self.physics[xinds, startinds]
        o2 = self.physics[xinds, endinds]
        a = self.actions[xinds, startinds]

        return (o1.cuda(), o2.cuda(), a.cuda())

    def insert_to_buffer(self, encodings, actions, physics):
        if isinstance(encodings, list):
            encodings = torch.stack(encodings)
            actions = torch.stack(actions)
            physics = torch.stack(physics)
        self.encodings[(self.cid%self.buffer_size)] = encodings
        self.actions[(self.cid%self.buffer_size)] = actions
        self.physics[(self.cid%self.buffer_size)] = physics
        self.cid = (self.cid + 1)

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

    def eval(self, states, width=.05):
        stats = []
        num_success = 0
        start_states = np.array(((-.2, .2), (.2,.2), (.2, -.2), (-.2,-.2)))
        offsets = sample_sphere(5, width)
        
        for start_state in start_states:
            success = False
            for offset in offsets:
                goal = start_state + offset
                self.train_env.reset()
                with self.train_env.physics.reset_context():
                    self.train_env.physics.set_state(np.r_[goal, np.zeros(2)])
                time_step = self.train_env.step((0,0))
                goal_emb = get_encoding(self.agent, time_step).squeeze()
                time_step = self.train_env.reset()
                with self.train_env.physics.reset_context():
                    self.train_env.physics.set_state(np.r_[start_state, np.zeros(2)])
                time_step = self.train_env.step((0,0))
                while not time_step.last():
                    encoding = get_encoding(self.agent, time_step).squeeze()
                    action = self.goal_agent.act(encoding, goal_emb, None, None, True)
                    time_step = self.train_env.step(action)
                    physics = time_step.physics
                    dist_to_goal = np.linalg.norm((goal - physics[:2]))
                    print(dist_to_goal)
                    if dist_to_goal < .01:
                        print("success!")
                        success = True
                        break
                if success:
                    num_success += 1

    def eval_bottomright(self, states, width=.05):
        stats = []
        num_success = 0
        success = False
        goal = (.2, -.2)
        self.train_env.reset()
        with self.train_env.physics.reset_context():
            self.train_env.physics.set_state(np.r_[goal, np.zeros(2)])
        time_step = self.train_env.step((0,0))
        goal_emb = get_encoding(self.agent, time_step).squeeze()
        time_step = self.train_env.reset()
        while not time_step.last():
            encoding = get_encoding(self.agent, time_step).squeeze()
            action = self.goal_agent.act(encoding, goal_emb, None, None, True)
            time_step = self.train_env.step(action)
            physics = time_step.physics
            dist_to_goal = np.linalg.norm((goal - physics[:2]))
            print(physics)
            if dist_to_goal < .01:
                print("success!")
                success = True
                break
        if success:
            num_success += 1

    def train(self):
        import IPython as ipy; ipy.embed(colors="neutral")
        num_eps = 1000
        batch_size = 256
        max_horizon = 20
        num_updates = 100
        self.goal_agent.actor.cuda()


        for ep in range(num_eps):
            #goal = np.random.sample(2) * .6 - .3
            goal = (.2, -.2)
            print(ep)
            encodings = []
            actions = []
            physics = []
            if False:
                rand = np.random.rand()
                startx, starty = np.random.rand(2) * 0.2
                if rand < 0.33:
                    startx = -startx - 0.1
                    starty = starty + 0.1
                elif rand < 0.66:
                    startx = startx + 0.1
                    starty = starty + 0.1
                else:
                    startx = -startx - 0.1
                    starty = -starty - 0.1
                start_state = (startx, starty)
                with self.train_env.physics.reset_context():
                    self.train_env.physics.set_state(np.r_[start_state, np.zeros(2)])
            self.train_env.reset()
            time_step = self.train_env.step((0, 0))
            encodings.append(get_encoding(self.agent, time_step).squeeze())
            physics.append(torch.tensor(time_step.physics))
            while not time_step.last():
                action = self.expert.act(time_step.physics, goal, None, None)
                actions.append(torch.tensor(action))
                time_step = self.train_env.step(action)
                encodings.append(get_encoding(self.agent, time_step).squeeze())
                physics.append(torch.tensor(time_step.physics))
            self.insert_to_buffer(encodings, actions, physics)

            if ep > 15:
                for _ in range(num_updates):
                    obs, goal_obs, desired_action = self.sample(batch_size, max_horizon)
                    res = self.goal_agent.update_actor(
                        obs, goal_obs, desired_action, max_horizon, ep
                    )
                print(res)

    def train2(self):
        import IPython as ipy; ipy.embed(colors="neutral")
        num_eps = 1000
        batch_size = 2048
        max_horizon = 5
        num_updates = 100000
        self.goal_agent.actor.cuda()
        with open("/home/maxgold/workspace/explore/proto_explore/url_benchmark/pretrain_dataset.pkl", "rb") as f:
            self.encodings, self.actions, self.physics = pickle.load(f)
        self.encodings = self.encodings[:700]
        self.actions = self.actions[:700]
        self.physics = self.physics[:700]
        self.cid = 700
        ep = 0


        for _ in range(num_updates):
            obs, goal_obs, desired_action = self.sample(batch_size, max_horizon)
            res = self.goal_agent.update_actor(
                obs, goal_obs, desired_action, max_horizon, ep
            )
            if _ % 50 == 0:
                print(res)
        print(res)

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f"snapshot_{self.global_frame}.pt"
        keys_to_save = ["agent", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot_base_dir = Path(self.cfg.snapshot_base_dir)
        if "point" in self.cfg.task:
            if "no_goal" in self.cfg.task:
                domain = self.cfg.task
                domain = re.sub("_v[1-9]", "", domain)
                print(f"LOADING FROM DOMAIN {domain}")
            else:
                domain = "point_mass_maze"
        else:
            domain, _ = self.cfg.task.split("_", 1)

        snapshot_dir = Path(f"models/pixels/{self.cfg.task}/proto_proto")

        def try_load(seed):
            snapshot = (
                Path("/home/maxgold/workspace/explore/proto_explore/url_benchmark")
                / snapshot_dir
                / str(seed)
                / f"snapshot_{self.cfg.snapshot_ts}.pt"
            )
            # import IPython as ipy; ipy.embed(colors='neutral')
            if not snapshot.exists():
                return None
            with snapshot.open("rb") as f:
                payload = torch.load(f)
            return payload

        # try to load current seed
        payload = try_load(self.cfg.seed)
        if payload is not None:
            return payload
        # otherwise try random seed
        while True:
            seed = np.random.randint(1, 11)
            payload = try_load(seed)
            if payload is not None:
                return payload
        return None


@hydra.main(config_path=".", config_name="finetune")
def main(cfg):
    from goal_policy import Workspace as W

    root_dir = Path.cwd()
    workspace = W(cfg)
    workspace.train2()


if __name__ == "__main__":
    main()

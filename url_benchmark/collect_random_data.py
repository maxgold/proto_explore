import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
import itertools

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
import random

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder
from torch.utils.data import IterableDataset
import scipy.spatial.distance as ssd

torch.backends.cudnn.benchmark = True

from dmc_benchmark import PRIMAL_TASKS
from collections import deque


def sample_action(env):
    shape = env.action_spec().shape
    minv = env.action_spec().minimum
    maxv = env.action_spec().maximum
    action = np.random.sample(shape) * (maxv - minv) + minv
    return action


def get_emb(agent, obs):
    obs = torch.tensor(obs).cuda()
    emb = agent.encoder(obs).reshape(-1)
    emb = agent.predictor(emb).cpu().detach().numpy()
    return emb


def test_agent(
    proto_agent,
    goal_agent,
    env,
    start2d,
    goal2d,
    num_eps=5,
    num_steps=100,
    use_expert=False,
):
    rewards1 = []
    rewards2 = []
    if start2d.shape[0] == 2:
        start = np.r_[start2d, np.zeros(2)]
    with env.physics.reset_context():
        env.physics.set_state(start)
    for _ in range(frame_stack):
        time_step = env.step((0, 0))
    start_emb = get_emb(proto_agent, time_step.observation)

    if goal2d.shape[0] == 2:
        goal = np.r_[goal2d, np.zeros(2)]
    with env.physics.reset_context():
        env.physics.set_state(goal)
    for _ in range(frame_stack):
        time_step = env.step((0, 0))
    goal_emb = get_emb(proto_agent, time_step.observation)

    for _ in range(num_eps):
        step = 0
        done = False
        with env.physics.reset_context():
            env.physics.set_state(start)
        for _ in range(frame_stack):
            time_step = env.step((0, 0))
        emb = get_emb(proto_agent, time_step.observation)
        # action = (0, 0)
        # time_step = env.step(action)
        # obs = time_step.observation
        while (step < num_steps) and (not done):
            if use_expert:
                action = agent.act(obs, goal, 0, True)
            else:
                try:
                    action = goal_agent.act(emb, goal_emb, {}, 0, True)
                except:
                    goal = np.r_[goal, np.zeros(2)]
                    action = goal_agent.act(obs, goal, {}, 0, True)

            time_step = env.step(action)
            cpos = time_step.physics
            # print(time_step.physics[:2])
            obs = time_step.observation
            emb = get_emb(proto_agent, obs)
            done = np.linalg.norm(cpos[:2] - goal2d[:2]) < 0.01
            step += 1
        rewards1.append(done)
        rewards2.append(step)
    return rewards1, rewards2


def get_clusters(obs, clusters):
    dists = ssd.cdist(obs, clusters)
    cluster_assgn = dists.argmin(1)
    return cluster_assgn


def choose_action(vd, N2, visit_count2, path2, c=1.4):
    vd = {v: d for v, d in vd.items() if v not in path2}
    if len(vd.keys()) == 0:
        return -1
    options = list(vd.keys())
    vals = list(vd.values())
    if (N2 == 0) or (sum(v for v in vals) == 0):
        action = np.random.choice(options)
    else:
        # p = []
        # for a, v in zip(options, vals):
        #    p.append(v[0] / (v[1]+1e-5)  + c*np.sqrt(N / (visit_count[a]+1e-5)))
        action = options[np.argmax(vals)]
    return action


class ReplayBuffer2(IterableDataset):
    def __init__(self, states, actions, max_horizon):
        self.states = states
        self.actions = actions
        self.max_horizon = max_horizon

    def sample(self):
        idx = np.random.randint(0, len(self.states) - self.max_horizon)
        horizon = np.random.randint(1, self.max_horizon + 1)
        obs = self.states[idx]
        goal = self.states[idx + horizon]  # [:2]
        action = self.actions[idx]
        return (obs, action, goal, horizon)

    def __len__(self):
        return len(self.states)

    def __iter__(self):
        while True:
            yield self.sample()


class ReplayBufferOnline(IterableDataset):
    def __init__(self, maxlen, max_horizon=20):
        from collections import deque

        self.states = deque(maxlen=maxlen)
        self.actions = deque(maxlen=maxlen)
        self.max_horizon = 20

    def sample1(self):
        idx = np.random.randint(0, len(self.states) - self.max_horizon)
        horizon = np.random.randint(1, self.max_horizon + 1)
        obs = self.states[idx]
        goal = self.states[idx + horizon]  # [:2]
        action = self.actions[idx]
        return (obs, action, goal, horizon)

    def sample(self, batch_size=32):
        obses = []
        actions = []
        goals = []
        for _ in range(batch_size):
            sample = self.sample1()
            obses.append(sample[0][None])
            actions.append(sample[1][None])
            goals.append(sample[2][None])

        obses = np.concatenate(obses, 0)
        actions = np.concatenate(actions, 0)
        goals = np.concatenate(goals, 0)
        return obses, actions, goals

    def __len__(self):
        return len(self.states)

    def __iter__(self):
        while True:
            yield self.sample()

    def seed(self, obs):
        self.states.append(obs)

    def insert(self, obs, action):
        self.states.append(obs)
        self.actions.append(action)


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(
    states,
    actions,
    batch_size,
    num_workers,
    max_horizon=20,
):
    iterable = ReplayBuffer2(states, actions, max_horizon)

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader


import IPython as ipy

ipy.embed(colors="neutral")

proto_agent = torch.load(
    "/home/maxgold/nina_proto_encoder/optimizer_proto_encoder1_1000000.pth"
)

task = "point_mass_maze_reach_bottom_right"
obs_type = "pixels"
frame_stack = 3
action_repeat = 5
seed = 1
time_limit = int(1e6)

env = dmc.make(task, obs_type, frame_stack, action_repeat, seed, time_limit=time_limit)

obs = torch.tensor(env.reset().observation).cuda()


proto_agent.encoder.cuda()
proto_agent.predictor.cuda()
obs = torch.tensor(env.reset().observation).cuda()


states = []
actions = []
num_steps = int(1e5)

time_step = env.reset()
if obs_type == "states":
    states.append(time_step.observation)
else:
    states.append(get_emb(proto_agent, time_step.observation))
    obs.cpu()

for _ in tqdm.tqdm(range(num_steps)):
    action = sample_action(env)
    actions.append(action)
    time_step = env.step(action)
    if obs_type == "states":
        states.append(time_step.observation)
    else:
        states.append(get_emb(proto_agent, time_step.observation))
    if time_step.last():
        break

import pickle

with open("tmp.pkl", "rb") as f:
    states, actions = pickle.load(f)

if False:
    with open("tmp.pkl", "wb") as f:
        pickle.dump([states, actions], f)

max_horizon = 20

loader = iter(make_replay_loader(states, actions, 512, 0, max_horizon=max_horizon))

from agent.gcsl import GCSLAgent2

goal_dim = (16,)
lr = 1e-3
hidden_dim = 256

goal_agent = GCSLAgent2(
    (16,),
    env.action_spec().shape,
    goal_dim,
    "cuda",
    lr,
    hidden_dim,
    max_horizon + 1,
    loss="cos",
)

logger = Logger(Path("./test2"), use_tb=False, use_wandb=False)

train_steps = int(5e4)
step = 0
while step < train_steps:
    metrics = goal_agent.update(loader, step)
    logger.log_metrics(metrics, step, ty="train")
    if step % 1000 == 0:
        print(step)
        print(metrics)
    # if step % 100 == 0:
    #    with logger.log_and_dump_ctx(step, ty="train") as log:
    #        log("fps", 1)
    step += 1

states2d = np.array([s for s in states])

clusters2d = np.arange(-0.29, 0.29, 0.05)
clusters2d = np.array(list(itertools.product(clusters2d, clusters2d)))


clusters2 = []
for c in tqdm.tqdm(clusters2d):
    if obs_type == "states":
        dists = ssd.cdist(c[None], states2d)
    else:
        # find the closest embedding to the state
        tmp = np.r_[c, np.zeros(2)]
        with env.physics.reset_context():
            env.physics.set_state(tmp)
        for _ in range(frame_stack):
            time_step = env.step((0, 0))

        emb = get_emb(proto_agent, time_step.observation)
        dists = ssd.cdist(emb[None], states2d)

    new_cluster = states2d[dists.argmin()]
    clusters2.append(new_cluster)

clusters = np.array(clusters2)


dists = []
stride = int(5e4)
total = len(states)
globals().update(locals())
for tstates in (states2d[i : i + stride] for i in range(0, total, int(stride))):
    dists.append(get_clusters(tstates, clusters))
cluster_assgn = np.concatenate(dists, 0)


offset = 100
shifted = cluster_assgn[offset:]

graph = {}
for i in range(clusters.shape[0]):
    inds = cluster_assgn[:-offset] == i
    vals = np.unique(shifted[inds], return_counts=True)
    graph[i] = {}
    for v, c in zip(*vals):
        if v != i:
            graph[i][v] = c

# d = torch.load(
#    "/home/maxgold/workspace/explore/proto_explore/url_benchmark/models/states/point_mass_maze_reach_bottom_right/proto/1/snapshot_1000000.pt"
# )
# agent = d["goal_agent"]
# goal_agent2 = torch.load(
#    "/home/maxgold/workspace/explore/proto_explore/output/2022.10.12/194502_gcac_gcsl_nohorizon/agent"
# )
# env = dmc.make(cfg.task, seed=0, goal=(0.25, -0.25))


pairwise_res = {}
list_res1 = []
list_res2 = []
list_res3 = []

for c1, v in tqdm.tqdm(graph.items()):
    print(np.mean(list_res2))
    if c1 >= 0:
        start = clusters2d[c1]
        pairwise_res[c1] = {}
        for c2 in v.keys():
            goal = clusters2d[c2]
            if np.linalg.norm(start - goal) < 0.25:
                # res1 = test_agent(expert, env, start, goal, use_expert=True)
                res2 = test_agent(
                    proto_agent,
                    goal_agent,
                    env,
                    start,
                    goal,
                    use_expert=False,
                    num_eps=3,
                )
                # res3 = test_agent(goal_agent2, env, start, goal, use_expert=False)
                pairwise_res[c1][c2] = res2
                # list_res1.append(np.mean(res1[0]))
                list_res2.append(np.mean(res2[0]))
        # list_res3.append(np.mean(res3[0]))

for c1, v in pairwise_res.items():
    for c2, (v1, v2) in v.items():
        if np.mean(v1) < 1:
            print(np.linalg.norm(clusters2d[c1] - clusters2d[c2]))

proto_agent_orig = torch.load(
    "/home/maxgold/nina_proto_encoder/optimizer_proto_encoder1_1000000.pth"
)
proto_agent = torch.load(
    "/home/maxgold/nina_proto_encoder/optimizer_proto_encoder1_1000000.pth"
)

cos_loss = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
dataset = ReplayBufferOnline(500)

dataset.seed(env.reset().observation)

optim = torch.optim.Adam(
    list(proto_agent.encoder.parameters()) + list(proto_agent.predictor.parameters()),
    lr=lr,
)

num_seed_steps = int(1e2)
for step in range(num_seed_steps):
    action = sample_action(env)
    actions.append(action)
    time_step = env.step(action)
    dataset.insert(time_step.observation, action)


num_goal_steps = int(1e3)

for _ in tqdm.tqdm(range(num_goal_steps)):
    action = sample_action(env)
    actions.append(action)
    time_step = env.step(action)
    dataset.insert(time_step.observation, action)

    batch = dataset.sample(32)
    obs, action, goal = batch
    proto_agent.encoder.parameters()

    emb = proto_agent.encoder(torch.tensor(obs).cuda())
    emb = proto_agent.predictor(emb)

    goal_emb = proto_agent.encoder(torch.tensor(goal).cuda())
    goal_emb = proto_agent.predictor(goal_emb)

    pred = goal_agent.actor(emb, goal_emb, None, 1).loc

    loss = -cos_loss(torch.tensor(action).cuda(), pred).mean()
    loss.backward()
    optim.step()
    optim.zero_grad()


# for each node we need to store number of wins when we've visited this node
# and the number of times we visted the node
# then at a mcts tree node i with child nodes (c1,c2,...,cn)
# the probability of choosing action j is w_j/n_j + c * sqrt(n_i/n_j)

time_limit = int(1e5)
action_repeat = 5
env = dmc.make(task, obs_type, frame_stack, action_repeat, seed, time_limit=time_limit)

# torch.save(goal_agent, "working_subgoal_agent.torch")


from agent.expert import ExpertAgent

expert = ExpertAgent()

num_mcts_eps = int(25)
episode_limit = int(1e3)


results = {}

for goal_ind in range(48, len(clusters)):
    mcts_tree = {i: {v: 0 for v in v2.keys()} for i, v2 in graph.items()}
    visit_count = {i: 0 for i in graph.keys()}
    fgoal = clusters[goal_ind]
    tres = []
    failed_paths = []
    success_paths = []

    use_expert = False

    for ep in range(num_mcts_eps):
        print(ep)
        steps = 0
        path = []
        pathxy = []
        time_step = env.reset()
        success = False
        while (steps < episode_limit) and not success:
            # print(path)
            # print(pathxy)
            obs = time_step.observation
            last_obs = obs
            cluster = get_clusters(obs[:2][None], clusters)[0]
            if cluster not in path:
                path.append(cluster)
            actions = mcts_tree[cluster]
            N = visit_count[cluster]
            proto_action = choose_action(actions, N, visit_count, path)
            if proto_action == -1:
                break
            goal = clusters[proto_action]
            goal = np.r_[goal, np.zeros(2)]
            # path.append(proto_action)
            pathxy.append(obs)
            done = False
            inner = 0
            while not done:
                if use_expert:
                    action = expert.act(obs[:2], goal[:2], 0, True)
                else:
                    action = agent.act(obs, goal, {}, 0, True)
                time_step = env.step(action)
                obs = time_step.observation
                if np.linalg.norm(obs[:2] - last_obs[:2]) < 1e-5:
                    break
                last_obs = obs
                done = np.linalg.norm(obs[:2] - goal[:2]) < 0.01
                inner += 1
                if np.linalg.norm(obs[:2] - fgoal) < 0.03:
                    success = True
                    print("SUCCESS")
                    break

            visit_count[cluster] += 1
        if success:
            rpath = list(reversed(path))
            for i, cluster in enumerate(rpath[:-1]):
                parent = rpath[i + 1]
                try:
                    mcts_tree[parent][cluster] += 1
                except:
                    pass
            success_paths.append(path)
        else:
            failed_paths.append(path)
        tres.append(success)
        if (len(tres) >= 4) and (np.mean(tres[-4:]) >= 1):
            break
    results[goal_ind] = tres
#        error = False
#        for x in path:
#            tmp = []
#            for p2 in success_paths:
#                if x in p2:
#                    print(f"Error on {x}")
#                    error = True
#                    break
#        if error:
#            break


# DONE
# 1. need a set of clusters
# 1a. assign each state to one of the clusters
# 2. need a mapping from each cluster to clusters reachable 5 steps away
# 3. create a subgoal policy (can just use ExpertAgent which I already have)

# TODO
# 1. need to figure out a way to train a local goal conditioned policy
# 1a. Is the action repeat of 5 an issue?
# 2. One option is to do GCSL

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs
from kdtree import KNN

import dmc
import utils
from logger import Logger
#from replay_buffer import make_replay_loader, make_replay_buffer
from video import VideoRecorder
import matplotlib.pyplot as plt
import seaborn as sns

torch.backends.cudnn.benchmark = True


def load_agent(path):
    return torch.load(path)


def eval(agent, env, num_eval_episodes, video_recorder, cfg, goal):
    step, episode, total_reward = 0, 0, 0
    eval_until_episode = utils.Until(num_eval_episodes)
    env = dmc.make(cfg.task, seed=cfg.seed, goal=goal)
    success = []
    perfs = []
    perfs2 = []
    while episode < num_eval_episodes:
        ok = False
        time_step = env.reset()
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(agent):
                # step doesn't matter here because eval_mode=True
                try:
                    action = agent.act(time_step.observation, goal, {}, 0, eval_mode=True)
                except:
                    action = agent.act(time_step.observation, goal, 0, eval_mode=True)
            time_step = env.step(action)
            total_reward += time_step.reward
            step += 1
            dist_to_goal = np.mean((env.physics.state()[:2] - goal) ** 2)
            if  dist_to_goal < .001:
                ok = True
                break
        success.append(int(ok))

        episode += 1
        dist_to_goal = -np.log(np.mean((env.physics.state()[:2] - goal) ** 2))
        perfs.append(total_reward)
        perfs2.append(dist_to_goal)
        total_reward = 0
        step = 0
    return success, perfs, perfs2


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    import IPython as ipy; ipy.embed(colors="neutral")
    work_dir = Path.cwd()
    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)
    #path_to_agent = "/home/maxgold/workspace/explore/proto_explore/output/2022.07.27/133447_gcac_gcsl_nohorizon/agent"
#    path_to_agent = "/home/maxgold/workspace/explore/proto_explore/output/2022.08.24/174419_gcac_gcsl_nohorizon/agent"
#    path_to_agent = "/home/maxgold/workspace/explore/proto_explore/output/2022.08.26/093304_gcac_gcsl_nohorizon/agent"
#    agent = load_agent(path_to_agent)
    d = torch.load(
        "/home/maxgold/workspace/explore/proto_explore/url_benchmark/models/states/point_mass_maze_reach_bottom_right/proto/1/snapshot_1000000.pt"
    )
    agent = d["goal_agent"]
    env = dmc.make(cfg.task, seed=0, goal=(0.25, -0.25))
    goal_grid = np.arange(-0.2, 0.2, 0.02)
    heatmapok = np.zeros((len(goal_grid), len(goal_grid)))
    heatmap1 = np.zeros((len(goal_grid), len(goal_grid)))
    heatmap2 = np.zeros((len(goal_grid), len(goal_grid)))
    num_eval_episodes = 10

    for i, x in enumerate(goal_grid):
        print(i)
        for j, y in enumerate(reversed(goal_grid)):
            if (abs(x) > 0.05) and (abs(y) > 0.05):
                goal = np.array((x, y))
                ok, perf, perf2 = eval(
                    agent,
                    env,
                    num_eval_episodes,
                    video_recorder,
                    cfg,
                    goal,
                )
                heatmapok[j,i] = np.mean(ok)
                heatmap1[j, i] = np.mean(perf)
                heatmap2[j, i] = np.mean(perf2)
    plt.clf()
    fig, ax = plt.subplots()
    sns.heatmap(heatmap1.T, cmap="Blues_r", cbar=False, ax=ax)
    plt.savefig(f"./heatmap1.png")

    plt.clf()
    fig, ax = plt.subplots()
    sns.heatmap(heatmap2, cmap="Blues_r", cbar=False, ax=ax)
    plt.savefig(f"./heatmap2.png")


if __name__ == "__main__":
    main()

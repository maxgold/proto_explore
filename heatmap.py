import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import hydra
import numpy as np
import numpy.random
import torch
from dm_env import specs
from kdtree import KNN
import matplotlib.pyplot as plt
import seaborn as sns

import dmc
import utils
from logger import Logger
from replay_buffer import make_replay_loader, make_replay_buffer
from video import VideoRecorder

torch.backends.cudnn.benchmark = True


def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]


def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1


GOAL_ARRAY = np.array([[-0.15, 0.15], [-0.15, -0.15], [0.15, -0.15], [0.15, 0.15]])


def eval(global_step, agent, env, logger, num_eval_episodes, video_recorder, cfg):
    step, episode, total_reward = 0, 0, 0
    eval_until_episode = utils.Until(num_eval_episodes)
    if cfg.goal:
        goal = np.random.sample((2,)) * 0.5 - 0.25
        env = dmc.make(cfg.task, seed=cfg.seed, goal=goal)
    while eval_until_episode(episode):
        time_step = env.reset()
        video_recorder.init(env, enabled=(episode == 0))
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(agent):
                if cfg.goal:
                    # goal = np.array((.2, .2))
                    action = agent.act(
                        time_step.observation, goal, global_step, eval_mode=True
                    )
                else:
                    action = agent.act(
                        time_step.observation, global_step, eval_mode=True
                    )
            time_step = env.step(action)
            video_recorder.record(env)
            total_reward += time_step.reward
            step += 1

        episode += 1
        video_recorder.save(f"{global_step}.mp4")

    with logger.log_and_dump_ctx(global_step, ty="eval") as log:
        log("episode_reward", total_reward / episode)
        log("episode_length", step / episode)
        log("step", global_step)


def eval_goal(global_step, agent, env, logger, video_recorder, cfg, goal):
    step, episode, total_reward = 0, 0, 0
    env = dmc.make(cfg.task, seed=cfg.seed, goal=goal)
    time_step = env.reset()
    video_recorder.init(env, enabled=True)
    while not time_step.last():
        with torch.no_grad(), utils.eval_mode(agent):
            if cfg.goal:
                # goal = np.array((.2, .2))
                action = agent.act(
                    time_step.observation, goal, global_step, eval_mode=True
                )
            else:
                action = agent.act(time_step.observation, global_step, eval_mode=True)
        time_step = env.step(action)
        video_recorder.record(env)
        total_reward += time_step.reward
        step += 1

    episode += 1
    # TODO: expand goal
    video_recorder.save(f"goal{global_step}:{str(goal)}.mp4")
    with logger.log_and_dump_ctx(global_step, ty="eval") as log:
        log("goal", goal)
        log("episode_reward", total_reward)
        log("episode_length", step)
        log("step", global_step)


def eval_random(env):
    time_step = env.reset()
    video_recorder.init(env, enabled=True)
    action_spec = env.action_spec()
    width = action_spec.maximum - action_spec.minimum
    base = action_spec.minimum
    while not time_step.last():
        action = width * np.random.sample(action_spec.shape) + base
        time_step = env.step(action)
        video_recorder.record(env)
        total_reward += time_step.reward
        step += 1

    episode += 1
    video_recorder.save(f"rand_episode.mp4")


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    env = dmc.make(cfg.task, seed=cfg.seed, goal=(0.25, -0.25))
    domain = get_domain(cfg.task)
    datasets_dir = work_dir / cfg.replay_buffer_dir
    import IPython as ipy ;ipy.embed(colors="neutral")
    #replay_dir = datasets_dir.resolve() / domain / cfg.expl_agent / "buffer"
#    replay_dir = Path(
#        "/home/maxgold/workspace/explore/proto_explore/url_benchmark/exp_local/2022.07.23/101256_proto/buffer2"
#    )
    replay_dir = Path(
        "/home/maxgold/workspace/explore/proto_explore/url_benchmark/exp_local/2022.08.11/202807_proto/buffer2"
    )
    replay_buffer = make_replay_buffer(
        env,
        replay_dir,
        cfg.replay_buffer_size,
        cfg.batch_size,
        0,
        cfg.discount,
        goal=cfg.goal,
        relabel=False,
    )
    states, actions, futures = replay_buffer.parse_dataset()
    states = states.astype(np.float64)
    knn = KNN(states[:, :2], futures)
    knn.query_k(np.array([0.15, 0.15])[None], 10)
    for i in range(5):
        start = i * 1000
        end = (i + 1) * 1000
        states, actions, futures = replay_buffer.parse_dataset(
            start_ind=start, end_ind=end
        )

        # for some reason i think the y coordinate gets flipped...
        heatmap, _, _ = np.histogram2d(states[:, 0], -states[:, 1], bins=50)
        # inds = ~((np.abs(states[:,0] + .05)<.03) * ((states[:,1] - .05)<.03))
        plt.clf()
        fig, ax = plt.subplots()
        sns.heatmap(np.log(1 + heatmap.T), cmap="Blues_r", cbar=False, ax=ax)
        plt.savefig(f"./heatmap_{start}_{end}.png")
    # sns.heatmap(np.log(-neg_data), cmap='Reds_r', cbar=False, ax=ax)
    plt.show()
    plt.clf()
    plt.hist2d(
        states[:, 0],
        states[:, 1],
        bins=[np.arange(-0.25, 0.25, 0.01), np.arange(-0.25, 0.25, 0.01)],
    )

    plt.show()
    heatmap, _, _ = np.histogram2d(states[:, 0], -states[:, 1], bins=50)
    # inds = ~((np.abs(states[:,0] + .05)<.03) * ((states[:,1] - .05)<.03))
    plt.clf()
    fig, ax = plt.subplots()
    sns.heatmap(np.log(1+heatmap.T), cmap="Blues_r", cbar=False, ax=ax)
    plt.savefig(f"./heatmap.png")

    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    fig, ax = plt.subplots()
    ax.imshow(heatmap.T, extent=extent, origin="lower")


if __name__ == "__main__":
    main()

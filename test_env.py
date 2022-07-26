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

import dmc
import utils
from logger import Logger
from replay_buffer import make_replay_loader
from video import VideoRecorder, Snapshot
import tqdm

torch.backends.cudnn.benchmark = True


def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]


def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1


def eval(agent, env, num_eval_episodes, video_recorder, cfg, goal):
    step, episode, total_reward = 0, 0, 0
    eval_until_episode = utils.Until(num_eval_episodes)
    env = dmc.make(cfg.task, seed=cfg.seed, goal=goal)
    video_recorder.init(env, enabled=(episode == 0))
    time_step = env.reset()
    while not time_step.last():
        with torch.no_grad(), utils.eval_mode(agent):
            # step doesn't matter here because eval_mode=True
            action = agent.act(
                time_step.observation, goal, 0, eval_mode=True
            )
        time_step = env.step(action)
        video_recorder.record(env)
        total_reward += time_step.reward
        step += 1
        if time_step.reward > .99:
            break

    video_recorder.save(f"expert_{str(goal)}.mp4")
    return time_step.reward


def eval_random(global_step, env, snapshot_recorder):
    time_step = env.reset()
    snapshot_recorder.init(env, enabled=True)
    action_spec = env.action_spec()
    width = action_spec.maximum - action_spec.minimum
    base = action_spec.minimum
    while not time_step.last():
        action = width * np.random.sample(action_spec.shape) + base
        time_step = env.step(action)
        video_recorder.record(env)

    snapshot_recorder.save(f"snapshot{global_step}.jpg")


@hydra.main(config_path=".", config_name="config")
def main(cfg):
    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # create logger
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

    work_dir = Path.cwd()
    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)
    env = dmc.make(cfg.task, seed=0, goal=(0.25, -0.25))
    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_shape=env.observation_spec().shape,
        action_shape=env.action_spec().shape,
        goal_shape=(2,),
    )
    import IPython as ipy; ipy.embed(colors="neutral")
    GOAL_ARRAY = np.array([[-0.15, 0.15], [-0.15, -0.15], [0.15, -0.15], [0.15, 0.15]])
    for goal in GOAL_ARRAY:
        reward = eval(
            agent,
            env,
            1,
            video_recorder,
            cfg,
            goal,
        )
        print(goal, reward)

        
    #snapper = Snapshot(work_dir if cfg.save_video else None)

    for ind in tqdm.tqdm(range(100)):
        goal = np.random.rand(2) * .5 - .25
        env = dmc.make(cfg.task, seed=cfg.seed, goal=goal)
        env.reset()
        snapper.snapshot(env, f"snapshot{ind}.jpg")

    eval_random(env, video_recorder)



if __name__ == "__main__":
    main()

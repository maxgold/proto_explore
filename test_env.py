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
    logger = Logger(work_dir, use_tb=cfg.use_tb)

        
    import IPython as ipy; ipy.embed(colors="neutral")
    snapper = Snapshot(work_dir if cfg.save_video else None)

    for ind in tqdm.tqdm(range(100)):
        goal = np.random.rand(2) * .5 - .25
        env = dmc.make(cfg.task, seed=cfg.seed, goal=goal)
        env.reset()
        snapper.snapshot(env, f"snapshot{ind}.jpg")

    eval_random(env, video_recorder)



if __name__ == "__main__":
    main()

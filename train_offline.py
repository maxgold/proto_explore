

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path
import random 
import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import make_replay_loader
from video import VideoRecorder
from replay_buffer import ndim_grid
import pandas as pd
from logger import save
import glob
torch.backends.cudnn.benchmark = True


def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]


def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1


def eval(global_step, agent, env, logger, num_eval_episodes, video_recorder, cfg):
    step, episode, total_reward = 0, 0, 0
    eval_until_episode = utils.Until(num_eval_episodes)
    if cfg.goal:
        goal = np.random.sample() * .5 - .25
        env = dmc.make(cfg.task, seed=cfg.seed, goal=goal)
    while eval_until_episode(episode):
        time_step = env.reset()
        video_recorder.init(env, enabled=(episode == 0))
        while not time_step.last():
            with torch.no_grad(), utils.eval_mode(agent):
                if cfg.goal:
                    #goal = np.array((.2, .2))
                    action = agent.act(time_step.observation, goal, global_step, eval_mode=True)
                else:
                    action = agent.act(time_step.observation, global_step, eval_mode=True)
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


def eval_goal(global_step, agent, env, logger, video_recorder, cfg, goal, model,work_dir):
    step, episode, total_reward = 0, 0, 0
    env = dmc.make(cfg.task, seed=None, goal=goal)
    time_step = env.reset()
    print(time_step.observation)
    #goal = np.array([np.random.sample() * -.25, np.random.sample() * .25])
    if cfg.eval==False:
        video_recorder.init(env, enabled=True)
        
    while not time_step.last():
        with torch.no_grad(), utils.eval_mode(agent):
            if cfg.goal:
                action = agent.act(time_step.observation, goal, global_step, eval_mode=True)
            else:
                action = agent.act(time_step.observation, global_step, eval_mode=True)
        time_step = env.step(action)
        if cfg.eval==False:
            video_recorder.record(env)
        total_reward += time_step.reward
        step += 1

    episode += 1
    if cfg.eval==False:
        video_recorder.save(f"goal{global_step}:{str(goal)}.mp4")
    if cfg.eval:
        print('saving')
        path = os.path.join(work_dir, 'eval_{}_{}.csv'.format(model.split('/')[-2], model.split('/')[-1].split('.')[-2]))
        save(path, [[goal, total_reward, time_step.observation[:2]]])
    else:
        with logger.log_and_dump_ctx(global_step, ty="eval") as log:
            log("goal", goal)
            log("episode_reward", total_reward)
            log("episode_length", step)
            log("steps", global_step)
        

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

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # create logger
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

    # create envs
    env = dmc.make(cfg.task, seed=cfg.seed, goal=(0.25, -0.25))

    # create agent
    if cfg.eval:
        print('evulating')
    elif cfg.distill:
        agent = hydra.utils.instantiate(
                                        cfg.agent,
                                        obs_shape=(4,),
                                        action_shape=(2,),
                                        goal_shape=(2,), 
                                        distill=cfg.distill
                                        )
    elif cfg.goal:
        agent = hydra.utils.instantiate(
            cfg.agent,
            obs_shape=env.observation_spec().shape,
            action_shape=env.action_spec().shape,
            goal_shape=(2,),
         )
    else:
        agent = hydra.utils.instantiate(
            cfg.agent,
            obs_shape=env.observation_spec().shape,
            action_shape=env.action_spec().shape,
        )
    #agent = torch.load(cfg.path)
    # create replay buffer
    data_specs = (
        env.observation_spec(),
        env.action_spec(),
        env.reward_spec(),
        env.discount_spec(),
    )

    # create data storage
    domain = get_domain(cfg.task)
    datasets_dir = work_dir / cfg.replay_buffer_dir
    if cfg.distill==False:
        replay_dir = datasets_dir.resolve() / domain / cfg.expl_agent / "buffer"
    else:
        replay_dir = datasets_dir.resolve()
    print(f"replay dir: {replay_dir}")
    #import IPython as ipy; ipy.embed(colors="neutral")

    replay_loader = make_replay_loader(
        env,
        replay_dir,
        cfg.replay_buffer_size,
        cfg.batch_size,
        cfg.replay_buffer_num_workers,
        cfg.discount,
        goal=cfg.goal,
        distill=cfg.distill
    )
    

    replay_iter = iter(replay_loader)
    # next(replay_iter) will give obs, action, reward, discount, next_obs
    
    # create video recorders
    video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

    timer = utils.Timer()

    global_step = 0

    train_until_step = utils.Until(cfg.num_grad_steps)
    eval_every_step = utils.Every(cfg.eval_every_steps)
    log_every_step = utils.Every(cfg.log_every_steps)
    print('ok')

    step=0

    while train_until_step(global_step):
        if cfg.eval:
            model_lst = glob.glob(str(cfg.path)+'*_40000.pth')
            if len(model_lst)>0:
                for ix in range(len(model_lst)):
                    print(ix)
                    agent = torch.load(model_lst[ix])
                     #logger.log("eval_total_time", timer.total_time(), global_step)
                    #if cfg.goal:
                    #import IPython as ipy; ipy.embed(colors="neutral")
                    goal_array = ndim_grid(2, 40)
                    #goal = np.array([np.random.sample() * -.25, np.random.sample() * -.25])
                    while step <3000:
                        for goal in goal_array:
                            print('evaluating', goal, 'model', model_lst[ix])
                        #import IPython as ipy; ipy.embed(colors="neutral")
                            eval_goal(global_step, agent, env, logger, video_recorder, cfg,goal, model_lst[ix], work_dir)
                                
                            step +=1
                            print(step)
                        
                    step=0
                    print(step)
        else:
            # try to evaluate
            if eval_every_step(global_step+1):
                logger.log("eval_total_time", timer.total_time(), global_step)
                if cfg.goal:
                    #goal = np.random.sample((2,)) * .5 - .25
                    #import IPython as ipy; ipy.embed(colors="neutral")
                    goal = np.random.sample((2,)) * .5 - .25
                    eval_goal(global_step, agent, env, logger, video_recorder, cfg, goal, goal, work_dir)
                else:
                    eval(global_step, agent, env, logger, cfg.num_eval_episodes, video_recorder, cfg)

            metrics = agent.update(replay_iter, global_step)
            logger.log_metrics(metrics, global_step, ty="train")
            if log_every_step(global_step):
                elapsed_time, total_time = timer.reset()
                with logger.log_and_dump_ctx(global_step, ty="train") as log:
                    log("fps", cfg.log_every_steps / elapsed_time)
                    log("total_time", total_time)
                    log("step", global_step)
        
            if global_step%10000==0:
                path = os.path.join(work_dir, 'optimizer_{}.pth'.format(global_step))
                torch.save(agent,path)

            global_step += 1


if __name__ == "__main__":
    main() 

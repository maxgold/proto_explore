import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils
from dm_control.utils import rewards


def classify_quadrant(state):
    if state[0] >= 0 and state[1] >= 0:
        return "topright"
    elif state[0] < 0 and state[1] >= 0:
        return "topleft"
    elif state[0] < 0 and state[1] < 0:
        return "bottomleft"
    elif state[0] >= 0 and state[1] < 0:
        return "bottomright"
    else:
        import IPython as ipy; ipy.embed(colors='neutral')
        import time; time.sleep(5)

go_right = np.array([1,0]).astype(np.float32)
go_up = np.array([0,1]).astype(np.float32)
go_left = np.array([-1,0]).astype(np.float32)
go_down = np.array([0,-1]).astype(np.float32)

gain = 10

def nav_to_topright(obs, goal):
    obs_quadrant = classify_quadrant(obs)
    obs = obs[:2]
    goal = goal[:2]
    if obs_quadrant == "topright":
        action = gain*(goal - obs)
    elif obs_quadrant == "topleft":
        if obs[1] > .2:
            action = go_right
        else:
            action = go_up
    elif obs_quadrant=="bottomleft":
        if obs[0] > -.2:
            action = go_left
        else:
            action = go_up
    elif obs_quadrant == "bottomright":
        if obs[0] > .2:
            action = go_up
        else:
            action = go_right
    return action

def nav_to_topleft(obs,goal):
    obs_quadrant = classify_quadrant(obs)
    obs = obs[:2]
    goal = goal[:2]
    if obs_quadrant == "topleft":
        action = gain*(goal - obs)
    elif obs_quadrant == "topright":
        if obs[1] > .2:
            action = go_left
        else:
            action = go_up
    elif obs_quadrant=="bottomleft":
        if obs[0] > -.2:
            action = go_left
        else:
            action = go_up
    elif obs_quadrant == "bottomright":
        if obs[0] > .2:
            action = go_up
        else:
            action = go_right
    return action

def nav_to_bottomleft(obs,goal):
    obs_quadrant = classify_quadrant(obs)
    obs = obs[:2]
    goal = goal[:2]
    if obs_quadrant == "bottomleft":
        action = gain*(goal - obs)
    elif obs_quadrant == "topright":
        if obs[1] > .2:
            action = go_left
        else:
            action = go_up
    elif obs_quadrant=="topleft":
        if obs[0] < -.2:
            action = go_down
        else:
            action = go_left
    elif obs_quadrant == "bottomright":
        if obs[1] > -.2:
            action = go_down
        else:
            action = go_left
    return action

def nav_to_bottomright(obs,goal):
    obs_quadrant = classify_quadrant(obs)
    obs = obs[:2]
    goal = goal[:2]
    if obs_quadrant == "bottomright":
        action = gain*(goal - obs)
    elif obs_quadrant == "topright":
        if obs[0] < .2:
            action = go_right
        else:
            action = go_down
    elif obs_quadrant=="topleft":
        if obs[0] < -.2:
            action = go_down
        else:
            action = go_left
    elif obs_quadrant == "bottomleft":
        if obs[1] > -.2:
            action = go_down
        else:
            action = go_right
    return action

class ExpertAgent:
    def __init__(self):
        self.training=False

    def train(self, training=True):
        pass

    def act(self, obs, goal, step, eval_mode):
        goal_quadrant = classify_quadrant(goal)
        obs_quadrant = classify_quadrant(obs)
        if goal_quadrant == "topright":
            return nav_to_topright(obs, goal)
        elif goal_quadrant == "topleft":
            return nav_to_topleft(obs, goal)
        elif goal_quadrant == "bottomleft":
            return nav_to_bottomleft(obs, goal)
        elif goal_quadrant == "bottomright":
            return nav_to_bottomright(obs, goal)
        else:
            import IPython as ipy; ipy.embed(colors='neutral')

    def init_meta(self):
        pass

    def update_meta(self):
        pass

    def update(self, replay_iter, step):
        pass

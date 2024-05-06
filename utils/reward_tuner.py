import time
import math
import torch
import numpy as np

def reward_tuner(reward_tune, reward, not_done, state = None, action = None,next_state = None):
    if reward_tune == 'normalize':
        reward = (reward - reward.mean()) / reward.std()
    elif reward_tune == 'iql_antmaze':
        reward = reward - 1.0
    elif reward_tune == 'iql_locomotion':
        reward = iql_normalize(reward, not_done)
    elif reward_tune == 'cql_antmaze':
        reward = (reward - 0.5) * 4.0
    elif reward_tune == 'antmaze':
        reward = (reward - 0.25) * 2.0
    elif reward_tune == 'antmaze_curiosity':
        reward = antmaze_curiosity(reward, not_done, state, action, next_state)
    return reward

def iql_normalize(reward, not_done):
	trajs_rt = []
	episode_return = 0.0
	for i in range(len(reward)):
		episode_return += reward[i]
		if not not_done[i]:
			trajs_rt.append(episode_return)
			episode_return = 0.0
	rt_max, rt_min = torch.max(torch.tensor(trajs_rt)), torch.min(torch.tensor(trajs_rt))
	reward /= (rt_max - rt_min)
	reward *= 1000.
	return reward

def antmaze_curiosity(reward, not_done, state, action, next_state):
    # reward shaping for antmaze, integrating ICM
    raise NotImplementedError
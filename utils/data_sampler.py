# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import time
import math
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional


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


class Data_Sampler(object):
    def __init__(self, data, device, reward_tune='no'):
        self.state = torch.from_numpy(data['observations']).float()
        self.action = torch.from_numpy(data['actions']).float()
        self.next_state = torch.from_numpy(data['next_observations']).float()
        reward = torch.from_numpy(data['rewards']).view(-1, 1).float()
        self.not_done = 1. - torch.from_numpy(data['terminals']).view(-1, 1).float()
        
        self.size = self.state.shape[0]
        self.state_dim = self.state.shape[1]
        self.action_dim = self.action.shape[1]
        self.device = device
        self.reward = reward_tuner(reward_tune, reward, self.not_done, self.state, self.action, self.next_state)
    
    def sample(self, batch_size):
        ind = torch.randint(0, self.size, size=(batch_size,))
        
        return (
			self.state[ind].to(self.device),
			self.action[ind].to(self.device),
			self.next_state[ind].to(self.device),
			self.reward[ind].to(self.device),
			self.not_done[ind].to(self.device)
		)
  
  
class Online_Sampler(object):
    def __init__(self, env, agent, batch_size, device, reward_tune='no'):
        self.agent = agent
        self.env = env
        self.device = device
        self.batch_size = batch_size
        self.reward_tune = reward_tune
        self.sample_number = 0
        self.episode_return = 0.0
        self.obs, self.done = self.env.reset(), False
        self.info = {}
        
    def sample(self, batch_size, **kwargs):
        states = []
        actions = []
        next_states = []
        rewards = []
        dones = []
        for _ in range(self.batch_size):
            batch_state = torch.from_numpy(
                np.expand_dims(self.obs, axis=0).astype("float32")
            )
            action = self.agent.sample_action(batch_state)
            next_state, reward, done, info = self.env.step(action)
            #print('state: ', next_state, 'reward: ', reward, 'done: ', self.done, 'info: ', info)#
            reward = reward_tuner(self.reward_tune, reward, 1-self.done, self.obs, action, next_state) # reward tuning
            self.episode_return = self.episode_return * (1 - self.done) + reward
            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            if info["TimeLimit.truncated"]:
                self.done = False
                
            states.append(self.obs)
            actions.append(action)
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)
            self.obs = next_state
            self.done = done
            self.info = info
            self.sample_number += 1
            
        return (
      		torch.tensor(states, dtype=torch.float32).to(self.device),
			torch.tensor(actions, dtype=torch.float32).to(self.device),
			torch.tensor(next_states, dtype=torch.float32).to(self.device),
			torch.tensor(rewards, dtype=torch.float32).to(self.device),
			torch.tensor(dones).to(self.device)
        )
    
    def get_sample_number(self):
        return self.sample_number
    
    
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
    

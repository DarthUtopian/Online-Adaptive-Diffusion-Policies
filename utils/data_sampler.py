# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import time
import math
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from utils.buffer import ReplayBuffer
from utils.reward_tuner import reward_tuner


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
  
  
class OffPolicySampler(object):
    def __init__(self, env, buffer_size, device, train_freq, reward_tune='no'):
        self.env = env
        self.device = device
        self.buffer_size = buffer_size
        self.reward_tune = reward_tune
        self.sample_number = 0
        self.obs, self.done = self.env.reset(), False
        self.info = {}
        self.buffer = ReplayBuffer(env, buffer_size, reward_tune)
        self.train_freq = train_freq # training steps
        
    def collect_rollouts(self, agent, **kwargs):
        t1 = time.time()#
        collect_step = 0
        while self._should_collect_more_steps(collect_step):
            batch_state = torch.from_numpy(
                np.expand_dims(self.obs, axis=0).astype("float32")
            )
            action = agent.sample_action(batch_state)
            next_state, reward, self.done, info = self.env.step(action)
            #print('state: ', next_state, 'reward: ', reward, 'done: ', self.done, 'info: ', info)#
            reward = reward_tuner(self.reward_tune, reward, 1-self.done, self.obs, action, next_state) # reward tuning
            if "TimeLimit.truncated" not in info.keys():
                info["TimeLimit.truncated"] = False
            if info["TimeLimit.truncated"]:
                self.done = False
        
            self.buffer.add(self.obs, action, next_state, reward, self.done)
            collect_step += 1
            self.info = info
            self.obs = next_state
            
            if self.done or info["TimeLimit.truncated"]:
                self.obs, self.info = self.env.reset(), {}
                
            self.sample_number += 1
            
        t2 = time.time()#
        sample_time = t2 - t1#
        #print("sample_time: ", sample_time)#
        
    def sample(self, batch_size):
        (batch_states, 
         batch_actions, 
         batch_next_states, 
         batch_rewards, 
         batch_dones) = self.buffer.get_samples(batch_size)
        return (torch.from_numpy(batch_states).float().to(self.device),
                torch.from_numpy(batch_actions).float().to(self.device),
                torch.from_numpy(batch_next_states).float().to(self.device),
                torch.from_numpy(batch_rewards).float().to(self.device),
                torch.from_numpy(batch_dones).float().to(self.device))
    
    def _should_collect_more_steps(self, collect_step) -> bool:
        if collect_step < self.train_freq:
            return True
        else:
            return False
        
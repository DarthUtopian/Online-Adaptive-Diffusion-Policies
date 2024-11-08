import time
import math
import torch
import numpy as np
import gym
from typing import Dict, List, Tuple, Union, Optional
from utils.reward_tuner import reward_tuner



class ReplayBuffer(object):
    def __init__(self, env, buffer_size:int=100000, reward_tune:str='no', obs_dim:int=None, action_dim:int=None):
        self.env = env
        self.obs_dim = obs_dim if obs_dim is not None else env.observation_space.shape[0]
        self.action_dim = action_dim if action_dim is not None else env.action_space.shape[0]
        self.buffer_size = buffer_size
        self.reward_tune = reward_tune
        self.episode_return = 0.0
        self.obs, self.done = self.env.reset(), False
        self.info = {}
        self.set_buffer()
        
    def set_buffer(self):
        self.buffer_states = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.buffer_actions = np.zeros((self.buffer_size, self.action_dim), dtype=np.float32)
        self.buffer_next_states = np.zeros((self.buffer_size, self.obs_dim), dtype=np.float32)
        self.buffer_rewards = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.buffer_dones = np.zeros((self.buffer_size, 1), dtype=np.float32)
        self.pos = 0
        self.full = False
        
    def add(self, state, action, next_state, reward, done):
        #TODO: adding batched rewards
        #print(state.shape, action.shape, next_state.shape, reward.shape, done.shape)#
        self.buffer_states[self.pos] = state
        self.buffer_actions[self.pos] = action
        self.buffer_next_states[self.pos] = next_state
        self.buffer_rewards[self.pos] = reward
        self.buffer_dones[self.pos] = done
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
            
    def add_batch(self, states, actions, next_states, rewards, dones):
        for i in range(len(states)):
            self.add(states[i], actions[i], next_states[i], rewards[i], dones[i])
        
    def get_samples(self, batch_size, **kwargs):
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
            
        return (
            self.buffer_states[batch_inds],
      		self.buffer_actions[batch_inds],
            self.buffer_next_states[batch_inds],
            self.buffer_rewards[batch_inds],
            self.buffer_dones[batch_inds]
        )
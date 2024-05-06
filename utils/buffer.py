import time
import math
import torch
import numpy as np
import gym
from typing import Dict, List, Tuple, Union, Optional
from utils.reward_tuner import reward_tuner



class ReplayBuffer(object):
    def __init__(self, env, buffer_size:int=100000, reward_tune:str='no'):
        self.env = env
        self.buffer_size = buffer_size
        self.reward_tune = reward_tune
        self.episode_return = 0.0
        self.obs, self.done = self.env.reset(), False
        self.info = {}
        self.set_buffer()
        
    def set_buffer(self):
        self.buffer_states = np.zeros((self.buffer_size, self.env.observation_space.shape[0]), dtype=np.float32)
        self.buffer_actions = np.zeros((self.buffer_size, self.env.action_space.shape[0]), dtype=np.float32)
        self.buffer_next_states = np.zeros((self.buffer_size, self.env.observation_space.shape[0]), dtype=np.float32)
        self.buffer_rewards = np.zeros((self.buffer_size), dtype=np.float32)
        self.buffer_dones = np.zeros((self.buffer_size), dtype=np.float32)
        self.pos = 0
        self.full = False
        
    def add(self, state, action, next_state, reward, done):
        self.buffer_states[self.pos] = state
        self.buffer_actions[self.pos] = action
        self.buffer_next_states[self.pos] = next_state
        self.buffer_rewards[self.pos] = reward
        self.buffer_dones[self.pos] = done
        
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0
        
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
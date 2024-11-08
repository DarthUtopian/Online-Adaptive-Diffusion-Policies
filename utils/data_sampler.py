# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import time
import math
import torch
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
from utils.buffer import ReplayBuffer
from utils.reward_tuner import reward_tuner
import gym
import copy


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
    def __init__(self, env, buffer_size, device, sample_batchsize, reward_tune='no', noise_params=None):
        self.env = env
        self.device = device
        self.buffer_size = buffer_size
        self.sample_batchsize = sample_batchsize # online data collecting size
        self.reward_tune = reward_tune #no
        self.sample_number = 0
        self.noise_params = noise_params
        
        if isinstance(self.env, gym.vector.AsyncVectorEnv):
            self.is_vector = True
            self.num_envs = self.env.num_envs
            assert self.sample_batchsize % self.num_envs == 0, (
                "sample_batchsize must be divisible by the number of environments"
            )
            self.horizon = self.sample_batchsize // self.num_envs
            self.obs_dim = self.env.single_observation_space.shape[0]
            self.action_dim = self.env.single_action_space.shape[0]
        else:
            self.is_vector = False
            self.num_envs = 1
            self.horizon = self.sample_batchsize
            self.obs_dim = self.env.observation_space.shape[0]
            self.action_dim = self.env.action_space.shape[0]
            
        self.obs, self.info = self.env.reset(), {}
        self.done = False
    
        if self.is_vector:
            # convert a dict of batched data to a list of dict of unbatched data
            # e.g. next_info = {"a": [1, 2, 3], "b": [4, 5, 6]} ->
            #      unbatched_infos = [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]
            # ref: https://stackoverflow.com/questions/5558418/list-of-dicts-to-from-dict-of-lists
            #self.info = [dict(zip(self.info, t)) for t in zip(*self.info.values())] if self.info else [{}] * self.num_envs
            self.info = [{}] * self.num_envs

        self.buffer = ReplayBuffer(env, buffer_size, reward_tune, obs_dim=self.obs_dim, action_dim=self.action_dim)
        
    def collect_rollouts(self, agent, **kwargs):
        t1 = time.time()#
        num_collected = 0
        while self._should_collect_more_steps(num_collected, **kwargs):
            batch_state = torch.from_numpy(
                np.expand_dims(self.obs, axis=0).astype("float32")
            ) if not self.is_vector else torch.from_numpy(
                self.obs.astype("float32")
            )
            if self.is_vector:
                action = agent.sample_action_batch(batch_state)
            else:
                action = agent.sample_action(batch_state)
            if self.noise_params is not None:
                action = self.add_noise(action)
            
            if self.is_vector:
                curr_obs = self.obs.copy()
                next_obs, reward, self.done, next_info = self.env.step(action)
                #print('state: ', next_obs, 'action: ', action, 'reward: ', reward, 'done: ', self.done, 'info: ', next_info)#
                reward = reward_tuner(self.reward_tune, reward, 1-self.done, self.obs, action, next_obs) # reward tuning
                self.obs = next_obs.copy()
                # For vector env, next_obs, reward, terminated, truncated, and next_info are batched data,
                # and vector env will automatically reset the environment when terminated or truncated is True,
                # So we need to get real final observation and info from next_info.
                unbatched_infos = next_info
                for i in range(self.num_envs):
                    if "final_observation" in next_info[i].keys():
                        next_obs[i, :] = np.stack(next_info[i]["final_observation"])
                    if "final_info" in next_info[i].keys():
                        unbatched_infos[i] = next_info[i]["final_info"]
                self.info = copy.deepcopy(unbatched_infos)
                self.buffer.add_batch(curr_obs, action, next_obs, reward.reshape(self.num_envs, 1), self.done.reshape(self.num_envs, 1))
                num_collected += self.num_envs
            else:
                action = action.reshape(-1)
                curr_obs = self.obs.copy()
                next_obs, reward, self.done, next_info = self.env.step(action)
                reward = reward_tuner(self.reward_tune, reward, 1-self.done, self.obs, action, next_obs) # reward tuning
                #print('state: ', next_obs, 'action: ', action, 'reward: ', reward, 'done: ', self.done, 'info: ', next_info)#
                # TODO: deprecate this after changing to gymnasium
                if "TimeLimit.truncated" not in next_info.keys():
                    next_info["TimeLimit.truncated"] = False
                if next_info["TimeLimit.truncated"]:
                    self.done = False
                
                self.obs = next_obs.copy()
                self.info = next_info
                if self.done or next_info["TimeLimit.truncated"]:
                    self.obs, self.info = self.env.reset(), {}
                    
                self.buffer.add(curr_obs, action, next_obs, reward.reshape(1), self.done.reshape(1))
                num_collected += 1
        
        self.sample_number += num_collected
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
                torch.from_numpy(1 - batch_dones).float().to(self.device)) #not_done

    def warmup(self, random_actor, warmup_steps):
        self.collect_rollouts(random_actor, is_warmup=True, warmup_steps=warmup_steps)
        
    def add_noise(self, action):
        noise = np.random.normal(self.noise_params["mean"], self.noise_params["std"], action.shape)
        action = action + noise
        return action
        
    def _should_collect_more_steps(self, num_collected, **kwargs) -> bool:
        is_warmup = kwargs.get("is_warmup", False)
        if is_warmup:
            return num_collected < kwargs.get("warmup_steps")
        else:
            return num_collected < self.sample_batchsize
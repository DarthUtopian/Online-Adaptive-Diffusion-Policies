import os
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt  
import gym
from typing import Dict, List, Tuple, Union, Optional
from utils import utils
from utils.logger import logger
from utils.visualization import plot_results
from utils.evaluation import eval_policy
from utils.data_sampler import Online_Sampler
from torch.utils.tensorboard import SummaryWriter



if __name__ == "__main__":
    env_name = 'hopper-medium-expert-v2'
    log_dir = 'results/hopper-medium-expert-v2|dist|diffusion-qg|T-5|lr_decay|ms-online|0'
    model_id = 1800
    
    seed = 100
    res = np.load(log_dir+"/eval.npy", allow_pickle=True)
    metric = {'eval_res': res[:, 0]}
    plot_results(metric, 'eval iter', 'reward', env_name, 'rewards', log_dir)
    
    env = gym.make(env_name)
    torch.manual_seed(0)
    np.random.seed(0)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    
    with open(os.path.join(log_dir, "variant.json"), 'r') as f:
        hyperparams = json.load(f)
        
    from agents.qg_diffusion import Diffusion_QL as Agent
    agent = Agent(state_dim=state_dim,
                    action_dim=action_dim,
                    max_action=max_action,
                    device='cuda' if torch.cuda.is_available() else 'cpu',
                    discount=hyperparams['discount'],
                    tau=hyperparams['tau'],
                    max_q_backup=hyperparams['max_q_backup'],
                    beta_schedule=hyperparams['beta_schedule'],
                    n_timesteps=hyperparams['T'],
                    eta=hyperparams['eta'],
                    lr=hyperparams['lr'],
                    lr_decay=hyperparams['lr_decay'],
                    lr_maxt=hyperparams['num_epochs'],
                    grad_norm=hyperparams['gn'])
    agent.load_model(dir=log_dir, id=model_id)
    
    t1 = time.time()
    eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(policy=agent, env_name=env_name, 
                                                                        seed=seed, eval_episodes=10, render_gif=True, 
                                                                        save_dir=log_dir)
    t2 = time.time()
    eval_time = (t2 - t1)
    utils.print_banner(f"policy evaluation finished, time{eval_time}")##
    
    print("reward: ", eval_res, "reward std: ", eval_res_std, "norm reward: ", eval_norm_res, "norm reward std: ", eval_norm_res_std)
  
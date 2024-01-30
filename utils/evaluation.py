import gym
import numpy as np
import torch

from utils import utils


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10, render_gif=False, save_dir=None):
    eval_env = gym.make(env_name)
    eval_env.seed(seed + 100)
    #eval_env.reset(seed=seed+100)
    frames = []
    
    scores = []
    for i in range(eval_episodes):
        print(f"evaluating episode {i}", end='\r', flush=True)
        traj_return = 0.
        state, done = eval_env.reset(), False
        while not done:
            if render_gif and i==0:
                frames.append(eval_env.render(mode='rgb_array'))
                
            action = policy.sample_action(np.array(state))
            state, reward, done, _ = eval_env.step(action)
            traj_return += reward
        scores.append(traj_return)

    avg_reward = np.mean(scores)
    std_reward = np.std(scores)

    normalized_scores = [eval_env.get_normalized_score(s) for s in scores]
    avg_norm_score = eval_env.get_normalized_score(avg_reward)
    std_norm_score = np.std(normalized_scores)

    if render_gif:
        assert len(frames) > 0
        assert save_dir is not None
        utils.make_gif(frames, save_dir, 'eval_render.gif')
        utils.print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f} render saved")
    else: 
        utils.print_banner(f"Evaluation over {eval_episodes} episodes: {avg_reward:.2f} {avg_norm_score:.2f}")
    
    return avg_reward, std_reward, avg_norm_score, std_norm_score
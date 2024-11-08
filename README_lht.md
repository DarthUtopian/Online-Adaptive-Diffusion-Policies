## Q-Guided Score Matching Diffusion Policie PyTorch Implementation
### Implementation based on:

**Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning**<br>
Zhendong Wang, Jonathan J Hunt and Mingyuan Zhou <br>
https://arxiv.org/abs/2208.06193 <br>

Abstract: *Offline reinforcement learning (RL), which aims to learn an optimal policy using a previously collected static dataset,
is an important paradigm of RL. Standard RL methods often perform poorly at this task due to the function approximation errors on
out-of-distribution actions. While a variety of regularization methods have been proposed to mitigate this issue, they are often
constrained by policy classes with limited expressiveness that can lead to highly suboptimal solutions. In this paper, we propose
representing the policy as a diffusion model, a recent class of highly-expressive deep generative models. We introduce Diffusion
Q-learning (Diffusion-QL) that utilizes a conditional diffusion model for behavior cloning and policy regularization. 
In our approach, we learn an action-value function and we add a term maximizing action-values into the training loss of the conditional diffusion model,
which results in a loss that seeks optimal actions that are near the behavior policy. We show the expressiveness of the diffusion model-based policy,
and the coupling of the behavior cloning and policy improvement under the diffusion model both contribute to the outstanding performance of Diffusion-QL.
We illustrate the superiority of our method compared to prior works in a simple 2D bandit example with a multimodal behavior policy.
We further show that our method can achieve state-of-the-art performance on the majority of the D4RL benchmark tasks for offline RL.*

## Experiments

### Requirements
Installations of [PyTorch](https://pytorch.org/), [MuJoCo](https://github.com/deepmind/mujoco), and [D4RL](https://github.com/Farama-Foundation/D4RL) are needed. Please see the ``requirements.txt`` for environment set up details.

### Running
Running experiments based our code could be quite easy, so below we use `walker2d-medium-expert-v2` dataset as an example. 

For the bandit toy experiments, run the code below. 
```
D4RL_SUPPRESS_IMPORT_ERROR=1 python bandit_toy.py --algo qgmb --env_name rings --exp qgmb_test_v3 --device 0 --T 5 --ms online --lr_decay --mode train --save_best_model

D4RL_SUPPRESS_IMPORT_ERROR=1 python bandit_toy.py --algo qgmb --env_name 8gaussians --exp qgmb_test_v2 --device 0 --T 5 --ms online --lr_decay --mode train --save_best_model

D4RL_SUPPRESS_IMPORT_ERROR=1 python bandit_toy.py --algo qgmb --env_name 8gaussians --exp qgmb_test_v3_05 --device 0 --T 100 --ms online --lr_decay --mode eval --save_best_model

D4RL_SUPPRESS_IMPORT_ERROR=1 python bandit_toy.py --algo qgmb --env_name swissroll --exp qgmb_test_v3 --device 0 --T 100 --ms online --lr_decay --mode eval --save_best_model
```

For reproducing the optimal results, we recommend running with 'online model selection' as follows. 
The best_score will be stored in the `best_score_online.txt` file.
```.bash
python main.py --env_name walker2d-medium-expert-v2 --device 0 --ms online --lr_decay
```
```
D4RL_SUPPRESS_IMPORT_ERROR=1 python main.py --env_name hopper-medium-expert-v2 --algo bc --exp bc --device 0 --T 5 --ms online --lr_decay --save_best_model

D4RL_SUPPRESS_IMPORT_ERROR=1 python main.py --env_name hopper-medium-replay-v2 --algo ql --exp ql --device 1 --T 20 --ms online --lr_decay --save_best_model

D4RL_SUPPRESS_IMPORT_ERROR=1 python main.py --env_name halfcheetah-medium-expert-v2 --algo edp --exp edp_td3 --device 1 --T 5 --ms online --lr_decay --save_best_model

D4RL_SUPPRESS_IMPORT_ERROR=1 python main.py --env_name Walker2d --exp qgmb_test --algo qgmb --device 1 --T 5 --ms online --lr_decay --save_best_model

D4RL_SUPPRESS_IMPORT_ERROR=1 python main.py --env_name halfcheetah-medium-v2 --exp newbase_newclamp2 --algo qg --device 0 --T 5 --ms online --lr_decay --save_best_model

D4RL_SUPPRESS_IMPORT_ERROR=1 python main.py --env_name kitchen-partial-v0 --exp test0_EBS_record --algo qg --device 1 --T 5 --ms online --lr_decay --save_best_model
```
For online tuning:
```
D4RL_SUPPRESS_IMPORT_ERROR=0 python main.py --env_name Humanoid-v2 --exp qgmb_new_largebuffer --algo qgmb --device 0 --T 5 --ms online --lr_decay --training_mode=online --pretrained_dir "results/halfcheetah-medium-expert-v2|test0_EBS_record|diffusion-qg|T-5|lr_decay|ms-online|10000" --seed 10000 --model_id 2000

D4RL_SUPPRESS_IMPORT_ERROR=0 python main.py --env_name Humanoid-v2 --exp qgmb_all_sample --algo qgmb --device 1 --T 5 --ms online --lr_decay --training_mode=online --pretrained_dir "results/halfcheetah-medium-expert-v2|test0_EBS_record|diffusion-qg|T-5|lr_decay|ms-online|10000" --seed 10000 --model_id 2000

D4RL_SUPPRESS_IMPORT_ERROR=0 python main.py --env_name halfcheetah-medium-replay-v2 --exp test0_pretrained_EBS_DEMO_fast1 --algo qg --device 0 --T 5 --ms online --lr_decay --training_mode=online --pretrained_dir "results/halfcheetah-medium-replay-v2|test0_EBS_record|diffusion-qg|T-5|lr_decay|ms-online|10000" --seed 1000 --model_id 2000

D4RL_SUPPRESS_IMPORT_ERROR=0 python main.py --env_name hopper-medium-v2 --exp test0_pretrained_EBS_DEMO_fast1_test --algo qg --device 0 --T 5 --ms online --lr_decay --training_mode=online --pretrained_dir "results/hopper-medium-v2|test0_EBS_record|diffusion-qg|T-5|lr_decay|ms-online|10000" --seed 10000 --model_id 1450
```
For conducting 'offline model selection', run the code below. The best_score will be stored in the `best_score_offline.txt` file.
```.bash
python main.py --env_name walker2d-medium-expert-v2 --device 0 --ms offline --lr_decay --early_stop
```

Hyperparameters for Diffusion-QL have been hard coded in `main.py` for easily reproducing our reported results. 
Definitely, there could exist better hyperparameter settings. Feel free to have your own modifications. 

## Citation

If you find this open source release useful, please cite in your paper:
```
@article{wang2022diffusion,
  title={Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning},
  author={Wang, Zhendong and Hunt, Jonathan J and Zhou, Mingyuan},
  journal={arXiv preprint arXiv:2208.06193},
  year={2022}
}
```


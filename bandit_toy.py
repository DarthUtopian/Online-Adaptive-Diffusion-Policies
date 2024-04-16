import torch
from torch.utils.data import DataLoader
from dataset.dataset import ToyDataset, energy_sample
import argparse
import gym
import d4rl
import numpy as np
import os
import torch
import json
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

from utils import utils
from utils.data_sampler import Data_Sampler
from utils.logger import logger, setup_logger
from torch.utils.tensorboard import SummaryWriter

hyperparameters = {
    "8gaussians": {
        "lr": 3e-4,
        "eta": 5.0,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 200,
        "gn": 2.0,
        "top_k": 1,
    },  # 5.0
    "swissroll": {
        "lr": 3e-4,
        "eta": 0.2,
        "max_q_backup": False,
        "reward_tune": "no",
        "eval_freq": 50,
        "num_epochs": 200,
        "gn": 2.0,
        "top_k": 1,
    },
}


def train(env, state_dim, action_dim, max_action, device, output_dir, args):
    # TODO: implement offline pre-training and online tunning pipline
    # Load buffer
    dataset = ToyDataset(name=env)
    data_dict = dataset.get_formatted_data(state_dim)
    data_sampler = Data_Sampler(data_dict, device, args.reward_tune)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    utils.print_banner("Loaded buffer")

    if args.algo == "ql":
        from agents.ql_diffusion import Diffusion_QL as Agent

        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=0,
            tau=args.tau,
            max_q_backup=args.max_q_backup,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            eta=args.eta,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_maxt=args.num_epochs,
            grad_norm=args.gn,
        )
    elif args.algo == "edp":
        from agents.edp_diffusion import Diffusion_QL as Agent

        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=0,
            tau=args.tau,
            max_q_backup=args.max_q_backup,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            eta=args.eta,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_maxt=args.num_epochs,
            grad_norm=args.gn,
        )
    elif args.algo == "qg":
        from agents.qg_diffusion import Diffusion_QL as Agent

        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=0,
            tau=args.tau,
            max_q_backup=args.max_q_backup,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            eta=args.eta,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_maxt=args.num_epochs,
            grad_norm=args.gn,
        )
    elif args.algo == "qgedm":
        from agents.qgedm_diffusion import Diffusion_QL as Agent

        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=0,
            tau=args.tau,
            max_q_backup=args.max_q_backup,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            eta=args.eta,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_maxt=args.num_epochs,
            grad_norm=args.gn,
        )
    elif args.algo == "awr":
        from agents.awr_diffusion import Diffusion_QL as Agent

        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=0,
            tau=args.tau,
            max_q_backup=args.max_q_backup,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            eta=args.eta,
            lr=args.lr,
            lr_decay=args.lr_decay,
            lr_maxt=args.num_epochs,
            grad_norm=args.gn,
        )
    elif args.algo == "bc":
        from agents.bc_diffusion import Diffusion_BC as Agent

        agent = Agent(
            state_dim=state_dim,
            action_dim=action_dim,
            max_action=max_action,
            device=device,
            discount=0,
            tau=args.tau,
            beta_schedule=args.beta_schedule,
            n_timesteps=args.T,
            lr=args.lr,
        )

    early_stop = False
    stop_check = utils.EarlyStopping(tolerance=1, min_delta=0.0)
    writer = SummaryWriter(output_dir)

    evaluations = []
    training_iters = 0
    max_timesteps = args.num_epochs * args.num_steps_per_epoch
    metric = 100.0
    utils.print_banner(f"Training Start", separator="*", num_star=90)
    while (training_iters < max_timesteps) and (not early_stop):
        iterations = int(args.eval_freq * args.num_steps_per_epoch)
        loss_metric = agent.train(
            data_sampler,
            iterations=iterations,
            batch_size=args.batch_size,
            log_writer=writer,
        )
        training_iters += iterations
        curr_epoch = int(training_iters // int(args.num_steps_per_epoch))

        # Logging
        utils.print_banner(f"Train step: {training_iters}", separator="*", num_star=90)
        logger.record_tabular("Trained Epochs", curr_epoch)
        logger.record_tabular("Actor Loss", np.mean(loss_metric["actor_loss"]))
        logger.record_tabular("Critic Loss", np.mean(loss_metric["critic_loss"]))
        logger.dump_tabular()

        # Evaluation
        # eval_res, eval_res_std, eval_norm_res, eval_norm_res_std = eval_policy(agent, args.env_name, args.seed,
        #                                                                       eval_episodes=args.eval_episodes)#, render_gif=True, save_dir=output_dir)

        evaluations.append(
            [
                np.mean(loss_metric["bc_loss"]),
                np.mean(loss_metric["ql_loss"]),
                np.mean(loss_metric["actor_loss"]),
                np.mean(loss_metric["critic_loss"]),
                curr_epoch,
            ]
        )
        np.save(os.path.join(output_dir, "eval"), evaluations)
        logger.dump_tabular()

        if args.save_best_model:
            agent.save_model(output_dir, curr_epoch)

    # Model Selection: online or offline


def eval(env, state_dim, action_dim, max_action, device, log_dir, args):
    model_id = 50#args.num_epochs
    with open(os.path.join(log_dir, "variant.json"), "r") as f:
        hyperparams = json.load(f)

    if args.algo == "ql":
        from agents.ql_diffusion import Diffusion_QL as Agent
    elif args.algo == "edp":
        from agents.edp_diffusion import Diffusion_QL as Agent
    elif args.algo == "qg":
        from agents.qg_diffusion import Diffusion_QL as Agent
    elif args.algo == 'qgedm':
        from agents.qgedm_diffusion import Diffusion_QL as Agent

    agent = Agent(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=device,
        discount=0,  # one-step rl
        tau=hyperparams["tau"],
        max_q_backup=hyperparams["max_q_backup"],
        beta_schedule=hyperparams["beta_schedule"],
        n_timesteps=hyperparams["T"],
        eta=hyperparams["eta"],
        lr=hyperparams["lr"],
        lr_decay=hyperparams["lr_decay"],
        lr_maxt=hyperparams["num_epochs"],
        grad_norm=hyperparams["gn"],
    )

    agent.load_model(dir=log_dir, id=model_id)
    plot_eval_action(
        policy=agent,
        model_id=model_id,
        tasks=[env],
        etas=[hyperparams["beta"]],
        save_dir=log_dir,
    )


def plot_bandit_scatter(save_dir, tasks, betas, show=False):
    for i, task in enumerate(tasks):
        plt.figure(figsize=(12, 3.0))
        axes = []
        G = gridspec.GridSpec(1, len(betas))
        for j, beta in enumerate(betas):
            plt.subplot(G[0, j])
            data, e = energy_sample(task, beta=beta, sample_per_state=1000)
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlim(-4.5, 4.5)
            plt.ylim(-4.5, 4.5)
            if j == 0:
                mappable = plt.scatter(
                    data[:, 0],
                    data[:, 1],
                    s=1,
                    c=e,
                    cmap="winter",
                    vmin=0,
                    vmax=1,
                    rasterized=True,
                )
                plt.yticks(ticks=[-4, -2, 0, 2, 4], labels=[-4, -2, 0, 2, 4])
            else:
                plt.scatter(
                    data[:, 0],
                    data[:, 1],
                    s=1,
                    c=e,
                    cmap="winter",
                    vmin=0,
                    vmax=1,
                    rasterized=True,
                )
                plt.yticks(
                    ticks=[-4, -2, 0, 2, 4], labels=[None, None, None, None, None]
                )
            axes.append(plt.gca())
            plt.xticks(ticks=[-4, -2, 0, 2, 4], labels=[-4, -2, 0, 2, 4])
            plt.title(f"beta={beta}")

        plt.tight_layout()
        plt.gcf().colorbar(mappable, ax=axes, fraction=0.1, pad=0.02, aspect=12)
        plt.savefig(save_dir + f"/scatter-data_{task}.png")

        if show:
            plt.show()
        else:
            plt.close()


def plot_eval_action(policy, model_id, tasks, etas, save_dir, show=False):
    x_range = torch.linspace(-4.5, 4.5, 90)
    y_range = torch.linspace(-4.5, 4.5, 90)
    a, b = torch.meshgrid(x_range, y_range, indexing="ij")
    id_mat = torch.stack([a, b], dim=-1)
    
    for i, task in enumerate(tasks):
        plt.figure(figsize=(7.0, 3.0))
        axes = []
        G = gridspec.GridSpec(1, 2*len(etas))
        for j, eta in enumerate(etas):    
            # -----plot value------
            plt.subplot(G[0, 2*j])
            states = np.zeros((90, 90, policy.state_dim))
            e = policy.critic.q_min( 
                id_mat.to(policy.device), 
                torch.from_numpy(states).float().to(policy.device)
                ).cpu().detach().numpy()  # sampled action
            vmin = e[25:65, 25:65].min()
            vmax = e[25:65, 25:65].max()
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlim(0, 89)
            plt.ylim(0, 89)
            if j == 0:
                mappable = plt.imshow(e[:,:,0], origin='lower', vmin=vmin, vmax=vmax, cmap='winter', rasterized=True)
                plt.yticks(ticks=[5, 25, 45, 65, 85], labels=[-4, -2, 0, 2, 4])
            else:
                plt.imshow(e, origin='lower', vmin=vmin, vmax=vmax, cmap='winter', rasterized=True)
                plt.yticks(
                    ticks=[5, 25, 65, 45, 85], labels=[None, None, None, None, None]
                )
            plt.xticks(ticks=[5, 25, 65, 45, 85], labels=[-4, -2, 0, 2, 4])
            axes.append(plt.gca())
            plt.title(f"eta={eta}")
            
            # -----plot sample-----
            plt.subplot(G[0, 2*j+1])
            states = np.zeros((1000, policy.state_dim))
            data, e = policy.sample(np.array(states))  # sampled action
            plt.gca().set_aspect("equal", adjustable="box")
            plt.xlim(-4.5, 4.5)
            plt.ylim(-4.5, 4.5)
            plt.scatter(
                data[:, 0],
                data[:, 1],
                s=1,
                rasterized=True,
            )
            plt.yticks(
                ticks=[-4, -2, 0, 2, 4], labels=[None, None, None, None, None]
            )
            plt.xticks(ticks=[-4, -2, 0, 2, 4], labels=[-4, -2, 0, 2, 4])
            axes.append(plt.gca())
            plt.title(f"eta={eta}")

        plt.tight_layout()
        plt.savefig(save_dir + f"/scatter-action_model{model_id}.png")
        cbar = plt.colorbar(mappable, ax=axes, fraction=0.1, pad=0.02, aspect=12)
        plt.gcf().axes[-1].yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        if show:
            plt.show()
        else:
            plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ### Experimental Setups ###
    parser.add_argument("--exp", default="exp_1", type=str)  # Experiment ID
    parser.add_argument(
        "--device", default=0, type=int
    )  # device, {"cpu", "cuda", "cuda:0", "cuda:1"}, etc
    parser.add_argument("--env_name", default="8gaussians", type=str)  # bandit tasks
    parser.add_argument("--beta", default=5.0, type=float)  # env energy
    parser.add_argument("--dir", default="results_toy", type=str)  # Logging directory
    parser.add_argument("--seed", default=0, type=int)  # Sets PyTorch and Numpy seeds
    parser.add_argument("--num_steps_per_epoch", default=1000, type=int)

    ### Optimization Setups ###
    parser.add_argument("--batch_size", default=256, type=int)
    parser.add_argument("--lr_decay", action="store_true")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--save_best_model", action="store_true")

    ### RL Parameters ###
    parser.add_argument("--tau", default=0.005, type=float)

    ### Diffusion Setting ###
    parser.add_argument("--T", default=5, type=int)
    parser.add_argument("--beta_schedule", default="vp", type=str)
    ### Algo Choice ###
    parser.add_argument(
        "--algo", default="ql", type=str
    )  # ['bc', 'ql', 'edp', 'qg', 'awr', 'qgedm']
    parser.add_argument(
        "--ms", default="offline", type=str, help="['online', 'offline']"
    )
    parser.add_argument("--mode", default="train", type=str, help="['train', 'eval']")

    args = parser.parse_args()
    args.device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    print("cuda", torch.cuda.is_available())
    args.output_dir = f"{args.dir}"

    args.num_epochs = hyperparameters[args.env_name]["num_epochs"]
    args.eval_freq = hyperparameters[args.env_name]["eval_freq"]
    args.eval_episodes = 10 if "v2" in args.env_name else 100

    args.lr = hyperparameters[args.env_name]["lr"]
    args.eta = hyperparameters[args.env_name]["eta"]
    args.max_q_backup = hyperparameters[args.env_name]["max_q_backup"]
    args.reward_tune = hyperparameters[args.env_name]["reward_tune"]
    args.gn = hyperparameters[args.env_name]["gn"]
    args.top_k = hyperparameters[args.env_name]["top_k"]

    # Setup Logging
    file_name = (
        f"{args.env_name}|beta-{args.beta}|{args.exp}|diffusion-{args.algo}|T-{args.T}"
    )
    if args.lr_decay:
        file_name += "|lr_decay"
    file_name += f"|ms-{args.ms}"

    if args.ms == "offline":
        file_name += f"|k-{args.top_k}"
    file_name += f"|{args.seed}"

    results_dir = os.path.join(args.output_dir, file_name)
    if args.mode == "eval":
        assert os.path.exists(results_dir), f"Model not found: {results_dir}"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    utils.print_banner(f"Saving location: {results_dir}")
    variant = vars(args)
    variant.update(version=f"Diffusion-Policies-RL")

    env = args.env_name  # def toy environment
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = 2
    action_dim = 2
    max_action = 4.5  # TODO: check this

    variant.update(state_dim=state_dim)
    variant.update(action_dim=action_dim)
    variant.update(max_action=max_action)
    setup_logger(os.path.basename(results_dir), variant=variant, log_dir=results_dir)
    utils.print_banner(f"Toy Env: {args.env_name}, action_dim: {action_dim}")

    # plot_bandit_scatter("results_toy", tasks=["8gaussians"], etas=[0, 1.0, 3.0, 5.0, 10.0])
    if args.mode == "train":
        if not os.path.exists(os.path.join(results_dir, f"scatter-data_{env}.png")):
            plot_bandit_scatter(results_dir, tasks=[env], betas=[args.beta])
        train(env, state_dim, action_dim, max_action, args.device, results_dir, args)
        eval(env, state_dim, action_dim, max_action, args.device, results_dir, args)
    elif args.mode == "eval":
        if not os.path.exists(os.path.join(results_dir, f"scatter-data_{env}.png")):
            plot_bandit_scatter(results_dir, tasks=[env], betas=[args.beta])
        eval(env, state_dim, action_dim, max_action, args.device, results_dir, args)

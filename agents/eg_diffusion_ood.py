import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils.logger import logger
from tqdm import trange, tqdm
from agents.diffusion import Diffusion
from agents.model import MLP
from agents.helpers import EMA, SinusoidalPosEmb


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(Critic, self).__init__()
        self.q1_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

        self.q2_model = nn.Sequential(nn.Linear(state_dim + action_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, hidden_dim),
                                      nn.Mish(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x), self.q2_model(x)

    def q1(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q1_model(x)

    def q2(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.q2_model(x)
    
    def q_min(self, state, action=None):
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


class Diffusion_EG(object):
    # TODO: implement OOD detection
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 discount,
                 tau,
                 max_q_backup=False,
                 eta=1.0,
                 beta_schedule='linear',
                 n_timesteps=100,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 lr_decay=False,
                 lr_maxt=1000,
                 grad_norm=1.0,
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        self.bc_model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        self.bc_actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.bc_model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps,).to(device)
        self.bc_actor_optimizer = torch.optim.Adam(self.bc_actor.parameters(), lr=lr)

        self.lr_decay = lr_decay
        self.grad_norm = grad_norm

        self.step = 0
        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.bc_actor_lr_scheduler = CosineAnnealingLR(self.bc_actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=0.)

        self.state_dim = state_dim
        self.max_action = max_action
        self.action_dim = action_dim
        self.discount = discount
        self.tau = tau
        self.eta = eta  # q_learning weight
        self.device = device
        self.max_q_backup = max_q_backup
        self.logp_thershold = 1.0 # enable conservative q learning
        self.num_updates = 1#64 #TODO: set this in kargs

    def step_ema(self):
        if self.step < self.step_start_ema:
            return
        self.ema.update_model_average(self.ema_model, self.actor)
        
    def train_bc(self, replay_buffer, iterations, batch_size=100, log_writer=None):
        # training behavior cloning diffusion
        metric = {'bc_loss': []}
        for i in tqdm(range(iterations), ncols=80):
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)
            """ BC Policy Training """
            bc_actor_loss = self.bc_actor.loss(action, state)  # bc training
            
            self.bc_actor_optimizer.zero_grad()
            bc_actor_loss.backward()
            if self.grad_norm > 0: 
                bc_actor_grad_norms = nn.utils.clip_grad_norm_(self.bc_actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('BC Actor Grad Norm', bc_actor_grad_norms.max().item(), self.step)
                log_writer.add_scalar('BC Actor Loss', bc_actor_loss.item(), self.step)

            metric['bc_loss'].append(bc_actor_loss.item())

        if self.lr_decay: 
            self.bc_actor_lr_scheduler.step()
            
        return metric

    def train(self, replay_buffer, iterations, batch_size=100, log_writer=None, train_mode='offline'):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for i in tqdm(range(iterations), ncols=80):
            # Sample replay buffer / batch
            if train_mode == 'online' and i % self.num_updates == 0:
                replay_buffer.collect_rollouts(self)
        
            state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

            """ Q Training """
            current_q1, current_q2 = self.critic(state, action)

            if self.max_q_backup:
                next_state_rpt = torch.repeat_interleave(next_state, repeats=10, dim=0)
                next_action_rpt = self.ema_model(next_state_rpt)
                target_q1, target_q2 = self.critic_target(next_state_rpt, next_action_rpt)
                target_q1 = target_q1.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q2 = target_q2.view(batch_size, 10).max(dim=1, keepdim=True)[0]
                target_q = torch.min(target_q1, target_q2)
            else:
                next_action = self.ema_model(next_state)
                target_q1, target_q2 = self.critic_target(next_state, next_action)
                target_q = torch.min(target_q1, target_q2)

            target_q = (reward + not_done * self.discount * target_q).detach()

            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
            
            ### OOD penalty ###
            with torch.no_grad():
                logp = self.bc_actor.logp_lower(next_action, next_state)
            penalty =  (- logp) > self.logp_thershold
            critic_loss += (penalty * (- logp - self.logp_thershold) * target_q).mean()
            
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()

            """ Policy Training """
            actor_loss, bc_loss = self.actor.loss_with_guidance(action, state, self.critic, self.eta)
            #actor_loss = self.actor.loss(action, state)  # bc testing
            #bc_loss = actor_loss
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            if self.grad_norm > 0: 
                actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.actor_optimizer.step()

            """ Step Target network """
            if self.step % self.update_ema_every == 0:
                self.step_ema()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            self.step += 1

            """ Log """
            if log_writer is not None:
                if self.grad_norm > 0:
                    log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                    log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
                #log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
                #log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
                log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
                log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

            metric['actor_loss'].append(actor_loss.item())
            metric['bc_loss'].append(bc_loss.item())
            #metric['ql_loss'].append(q_loss.item())
            metric['critic_loss'].append(critic_loss.item())

        if self.lr_decay: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=1, dim=0) #50
        with torch.no_grad():
            action = self.actor.sample(state_rpt)
            q_value = self.critic_target.q_min(state_rpt, action).flatten()
            idx = torch.multinomial(F.softmax(q_value), 1)
        return action[idx].cpu().data.numpy().flatten()
    
    def sample(self, state):
        # batched states
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action = self.actor.sample(state)
            q_value = self.critic_target.q_min(state, action)
        return action.cpu().data.numpy(), q_value.cpu().data.numpy()
    
    def evaluate_diff_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        state_rpt = torch.repeat_interleave(state, repeats=self.actor.n_timesteps+1, dim=0) #50
        with torch.no_grad():
            action, diffused_act = self.actor.sample(state, return_diffusion=True)
            q_values = self.critic_target.q_min(state_rpt, diffused_act.squeeze(1))
        return action.cpu().data.numpy().flatten(), q_values.cpu().data.numpy().flatten(), diffused_act.cpu().data.numpy()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')
            torch.save(self.critic.state_dict(), f'{dir}/critic.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))
            self.critic.load_state_dict(torch.load(f'{dir}/critic.pth'))
            
    def save_bc_model(self, dir, id=None):
        if id is not None:
            torch.save(self.bc_actor.state_dict(), f'{dir}/bc_actor_{id}.pth')
        else:
            torch.save(self.bc_actor.state_dict(), f'{dir}/bc_actor.pth')
            
    def load_bc_model(self, dir, id=None):
        if id is not None:
            self.bc_actor.load_state_dict(torch.load(f'{dir}/bc_actor_{id}.pth'))
        else:
            self.bc_actor.load_state_dict(torch.load(f'{dir}/bc_actor.pth'))
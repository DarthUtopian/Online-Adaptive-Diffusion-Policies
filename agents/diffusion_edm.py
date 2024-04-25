import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter, ModuleList
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as distributions
import pdb
import gc

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.helpers import (cosine_beta_schedule,
                            linear_beta_schedule,
                            vp_beta_schedule,
                            extract,
                            Losses)
from utils.utils import Progress, Silent


class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, max_action,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        
        sigmas_flex = torch.sqrt(betas)
        sigmas_inflex = torch.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alphas_cumprod[i-1]) / (1 - alphas_cumprod[i])) * betas[i]
        sigmas_inflex = torch.sqrt(sigmas_inflex)
        self.P_mean = -1.2
        self.P_std = 1.2
        self.sigma_data = 1.0

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()
        
    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas
        
    # ------------------------------------------ sampling ------------------------------------------#

    def forward_denoiser(self, x, state, sigma):
        x = x.to(torch.float32)
        # D_theta
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        F_x = self.model(c_in.reshape(-1,1) * x, sigma, state) # t = sigma ** 2
        D_x = c_skip.reshape(-1,1) * x + c_out.reshape(-1,1) * F_x
        return D_x
    
    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        raise NotImplementedError

    # @torch.no_grad()
    def p_sample(self, x, t, s):
        raise NotImplementedError

    # @torch.no_grad()
    def sample(self, state, num_steps=5, 
               sigma_min=0.002, sigma_max=2, rho=0.2, S_churn=5, S_min=0, S_max=float('inf'), S_noise=0.1,
               *args, **kwargs):
        device = self.betas.device
        batch_size = state.shape[0]
        step_indices = torch.arange(num_steps, dtype=torch.float64, device=state.device)
        t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
        t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])]) # t_N = 0
        
        return_diffusion = kwargs.get('return_diffusion', False)
        x_next = torch.randn(batch_size, self.action_dim, device=state.device) * t_steps[0]
        if return_diffusion: diffusion = [x_next]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])): # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            #gamma = (np.sqrt(2) - 1) * 1 if S_min <= t_cur <= S_max else 0
            t_hat = torch.as_tensor(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * torch.randn_like(x_cur)

            # Euler step.
            timesteps = torch.full((batch_size,), t_hat, device=device, dtype=torch.long)
            denoised = self.forward_denoiser(x_hat, state, timesteps)
            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                timesteps = torch.full((batch_size,), t_next, device=device, dtype=torch.long)
                denoised = self.forward_denoiser(x_next, state, timesteps)
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)
                
            if return_diffusion: diffusion.append(x_next)#

        action = x_next
        if return_diffusion:
            return action.clamp_(-self.max_action, self.max_action), torch.stack(diffusion, dim=1)
        else:
            return action.clamp_(-self.max_action, self.max_action)


    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )
        return sample

    def p_losses(self, x_start, state, sigma, weights=1.0):
        weights = weights * ((sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2)
        weights = weights.reshape(-1,1)
        noise = torch.randn_like(x_start) * sigma.reshape(-1,1)
        x_noisy = x_start + noise
        x_recon = self.forward_denoiser(x_noisy, state, sigma)

        assert x_start.shape == x_recon.shape
        #print("weights: ", weights)
        loss = self.loss_fn(x_recon, x_start, weights)
        #print("loss: ", loss)
        return loss
     
    def p_losses_with_guidance(self, x_start, state, value_func, eta, sigma, weights=1.0):
        # TODO: implement the guidance loss
        weights = weights * ((sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2)
        weights = weights.reshape(-1,1)
        noise = torch.randn_like(x_start) * sigma.reshape(-1,1)
        x_noisy = x_start + noise
        x_recon = self.forward_denoiser(x_noisy, state, sigma)
        
        assert x_start.shape == x_recon.shape
        
        x_start = x_start.requires_grad_()
        q1, q2 = value_func(state, x_start)
        q_score = torch.autograd.grad(outputs=torch.sum(torch.min(q1, q2)), inputs=x_start)[0]
        guidance = torch.clamp(q_score, -1, 1)
        #print(f"{t} guidance_shape: ", guidance.shape)#
        #print("guidance: ", guidance)#
        rec_loss = self.loss_fn(x_recon.clone().detach(), x_start.clone().detach(), weights)
        loss = self.loss_fn(x_recon, x_start + eta * guidance, weights)
        return loss, rec_loss

    def loss(self, x, state, weights=1.0, **kwargs):
        batch_size = len(x)
        rnd_normal = torch.randn(batch_size,).to(x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        return self.p_losses(x, state, sigma, weights)

    def loss_with_guidance(self, x, state, value_func, eta, weights=1.0, **kwargs):
        batch_size = len(x)
        rnd_normal = torch.randn(batch_size,).to(x.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        return self.p_losses_with_guidance(x, state, value_func, eta, sigma, weights)
    
    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)
    
    



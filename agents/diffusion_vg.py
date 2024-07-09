# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from agents.helpers import (
    cosine_beta_schedule,
    linear_beta_schedule,
    vp_beta_schedule,
    extract,
    Losses,
)
from utils.utils import Progress, Silent


class Diffusion(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        model,
        max_action,
        beta_schedule="linear",
        n_timesteps=100,
        loss_type="l2",
        clip_denoised=True,
        predict_epsilon=True,
    ):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model

        if beta_schedule == "linear":
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == "vp":
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer("betas", betas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod)
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1)
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer("posterior_variance", posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            torch.log(torch.clamp(posterior_variance, min=1e-20)),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod),
        )

        self.loss_fn = Losses[loss_type]()

        self.improve = False  # TODO: implement the improvement flag

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        """
        if self.predict_epsilon, model output is (scaled) noise;
        otherwise, model predicts x0 directly
        """
        if self.predict_epsilon:
            return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False, **kwargs):
        device = self.betas.device
        batch_size = shape[0]
        
        start_points = kwargs.get("start_points", None)
        if start_points is not None:
            assert start_points.shape == shape
            x = start_points # given start points
        else:
            x = torch.randn(shape, device=device)

        if return_diffusion:
            diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)
            progress.update({"t": i})
            if return_diffusion:
                diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=0)
        else:
            return x

    def p_sample_approximate(
        self, state, action, shape, verbose=False, return_diffusion=False, edp=True
    ):
        # EDP sampling, one step to approximate the action
        device = self.betas.device
        batch_size = shape[0]

        if return_diffusion:
            diffusion = [action]

        batch_size = len(action)

        t = torch.randint(
            0, self.n_timesteps, (batch_size,), device=action.device
        ).long()
        x_noisy = self.q_sample(x_start=action, t=t)
        x_approx = self.predict_start_from_noise(
            x_t=x_noisy, t=t, noise=self.model(x_noisy, t, state)
        )

        if return_diffusion:
            diffusion.append(x_approx)

        # print("original action: ", action, "approximate action: ", x_approx)
        if return_diffusion:
            return x_approx, torch.stack(diffusion, dim=1)
        else:
            return x_approx
    
    
    # @torch.no_grad()
    def ddim_sample(self, state, shape, verbose=False, return_diffusion=False, **kwargs):
        device = self.betas.device
        batch_size = shape[0]
        start_points = kwargs.get("start_points", None)
        
        if start_points is not None:
            assert start_points.shape == shape
            x = start_points # given start points
        else:
            x = torch.randn(shape, device=device)

        if return_diffusion:
            diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)
            progress.update({"t": i})
            if return_diffusion:
                diffusion.append(x)
                
        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=0)
        else:
            return x
        

    # @torch.no_grad()
    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        if "return_diffusion" in kwargs and kwargs["return_diffusion"]:
            action, diffused_act = self.p_sample_loop(state, shape, *args, **kwargs)
            return action, diffused_act

        if "edp" in kwargs:
            assert "action" in kwargs
            action = self.p_sample_approximate(
                state=state, shape=shape, *args, **kwargs
            )
        else:
            action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-self.max_action, self.max_action)

    def guided_sample(self, state, value_func, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.guided_sample_RED(state, shape, value_func, *args, **kwargs)
        return action.clamp_(-self.max_action, self.max_action)

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss
    
    def logp_lower(self, x, state, weights=1.0, **kwargs):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x_start=x, t=t, noise=noise)
        x_recon = self.model(x_noisy, t, state)
        assert noise.shape == x_recon.shape
        logp = - F.mse_loss(x_recon, noise, reduction='none').sum(dim=-1)
        
        return_predict = kwargs.get("ret_pred", False)
        if return_predict:
            x_start_mean = self.predict_start_from_noise(x_t=x_noisy, t=t, noise=x_recon)
            return logp, x_start_mean
        else:
            return logp

    def p_losses_with_guidance_pred(
        self, x_start, state, value_func, eta, t, weights=1.0
    ):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_noisy = x_noisy.requires_grad_()  # enable gradient computation
        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            x_start_mean = self.predict_start_from_noise(
                x_t=x_noisy, t=t, noise=x_recon.detach().clone()
            )
            q1, q2 = value_func(state, x_start_mean)
            # print("q1: ", q1, "q2: ", q2)
            """
            if np.random.uniform() > 0.5:
                q_loss = torch.min(q1, q2).sum() / q2.abs().mean().detach()
            else:
                q_loss = torch.min(q1, q2).sum() / q1.abs().mean().detach()
            q_score = torch.autograd.grad(outputs=q_loss, inputs=x_noisy)[0]
            """
            q_score = torch.autograd.grad(
                outputs=torch.sum(torch.min(q1, q2)), inputs=x_noisy
            )[0]
            SNR_t = extract(self.alphas_cumprod, t, x_start.shape) / (
                1 - extract(self.alphas_cumprod, t, x_start.shape)
            )
            # weights_guide = torch.min(SNR_t, 5) / SNR_t # TODO: implement the SNR weighting

            guidance = torch.clamp(
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * q_score,
                -1,
                1,
            )
            # print(f"{t} guidance_shape: ", guidance.shape)
            # print("guidance: ", guidance)
            rec_loss = self.loss_fn(
                x_recon.clone().detach(), noise.clone().detach(), weights
            )
            """
            if rec_loss <= 0.1:
                self.improve = True
                print("start improving the policy by Q")
            if rec_loss > 0.1 and not self.improve:
                eta = eta * 1e-3
            """
            loss = self.loss_fn(x_recon, noise - eta * guidance, weights)
        else:
            raise NotImplementedError

        return loss, rec_loss

    def p_losses_with_guidance_PIR(self, x_start, state, value_func, normal_q, eta, t, weights=1.0):
        ## this guidance loss is used for initial Diffusion-QL ##
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            # test0
            x_0 = x_start.clone().detach().requires_grad_()
            q1, q2 = value_func(state, x_0)
            q_score = torch.autograd.grad(outputs=torch.sum(torch.min(q1, q2)), inputs=x_0)[0].clone().detach()
            """
            x_start_mean = self.predict_start_from_noise(x_t=x_noisy, t=t, noise=x_recon.detach().clone())
            x_start_mean = x_start_mean.requires_grad_()
            q1, q2 = value_func(state, x_start_mean)
            q_loss = torch.min(q1, q2).sum() / normal_q
            q_score = torch.autograd.grad(outputs=q_loss, inputs=x_start_mean)[0]
            #print("normal_q: ", normal_q)#
            #SNR_t = extract(self.alphas_cumprod, t, x_start.shape) / (1 - extract(self.alphas_cumprod, t, x_start.shape))
            #ratio = extract(
            #    self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            #) / extract(self.sqrt_alphas_cumprod, t, x_start.shape) # x0_mean
            ratio = 1 / extract(self.sqrt_alphas_cumprod, t, x_start.shape) #new_base
            #ratio = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) ** 3 / extract(self.sqrt_alphas_cumprod, t, x_start.shape) #x0_mean_new
            #ratio = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) ** 2 #snr_test
            #ratio = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * extract(self.sqrt_alphas_cumprod, t, x_start.shape) #snr_real
            """
            ratio = extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            ) / extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            #ratio = 1 / extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            q_score_norm = torch.linalg.norm(q_score, dim=-1, keepdim=True)
            guidance = torch.clamp(ratio * q_score, -self.max_action, self.max_action)
            x_start_mean = self.predict_start_from_noise(x_t=x_noisy, t=t, noise=x_recon)
            
            rec_loss = torch.tensor([0.0]).to(x_start.device)
            #weights = weights / torch.clamp(q_score_norm.detach().clone(), 1.0, 10.0)#
            #print("q score norm: ", q_score_norm.reshape(-1))
            #print("guidance norm: ", torch.linalg.norm(guidance, dim=-1))
            loss = self.loss_fn(x_start_mean, 
                                torch.clamp(x_start + eta * guidance, -self.max_action, self.max_action), 
                                weights)
        else:
            raise NotImplementedError

        return loss, rec_loss

    def p_losses_with_guidance(
        self, x_start, state, value_func, eta, t, weights=1.0
    ):
        ## this guidance loss is tested for new vgdp 2024.7.9 ##
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            x_start_mean = self.predict_start_from_noise(x_t=x_noisy, t=t, noise=x_recon.detach().clone())
            x_start_mean_clip = torch.clamp(x_start_mean, -self.max_action, self.max_action).requires_grad_()
            q1, q2 = value_func(state, x_start_mean_clip)
            #if np.random.uniform() > 0.5:
            #    q_loss = q1.sum() / q2.abs().mean().detach().clone()
            #else:
            #    q_loss = q2.sum() / q1.abs().mean().detach().clone()
            q_loss = torch.min(q1, q2).sum()
            q_score = torch.autograd.grad(outputs=q_loss, inputs=x_start_mean_clip)[0]
            q_score_norm = torch.linalg.norm(q_score, dim=-1, keepdim=True).detach().clone()
            #print("\nq 1:", q1.mean())#
            #print("normal_q: ", q_score_norm.reshape(-1))#
            # SNR_t = extract(self.alphas_cumprod, t, x_start.shape) / (1 - extract(self.alphas_cumprod, t, x_start.shape))
            #ratio = extract(
            #    self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            #) / extract(self.sqrt_alphas_cumprod, t, x_start.shape) # x0_mean
            ratio = 1.0 / extract(self.sqrt_alphas_cumprod, t, x_start.shape) #new_base
            #ratio = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) ** 3 / extract(self.sqrt_alphas_cumprod, t, x_start.shape) #x0_mean_new
            #ratio = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) ** 2 #snr_test
            #ratio = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * extract(self.sqrt_alphas_cumprod, t, x_start.shape) #snr_real
            #print("t: ", t)
            #print("guidance: ", ratio * q_score)
            guidance = torch.clamp(ratio * q_score, -1.0, 1.0) #(-1,1)
            rec_loss = torch.tensor([0.0]).to(x_start.device)
            loss = self.loss_fn(x_recon, noise - eta * guidance, weights)# TODO
        else:
            raise NotImplementedError

        return loss, rec_loss

    def loss(self, x, state, weights=1.0, **kwargs):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def loss_with_guidance(self, x, state, value_func, eta, weights=1.0, **kwargs):
        batch_size = len(x)
        normal_q = kwargs.get("normal_q", 1.0)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses_with_guidance(x, state, value_func, normal_q, eta, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)
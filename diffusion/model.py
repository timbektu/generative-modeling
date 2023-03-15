import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from utils import (
    cosine_beta_schedule,
    default,
    extract,
    unnormalize_to_zero_to_one,
)
from einops import rearrange, reduce

class DiffusionModel(nn.Module):
    def __init__(
        self,
        model,
        timesteps = 1000,
        sampling_timesteps = None,
        ddim_sampling_eta = 1.,
    ):
        super(DiffusionModel, self).__init__()

        self.model = model
        self.channels = self.model.channels
        self.device = torch.cuda.current_device()
        # self.device =  torch.device("cpu") #TODO: change

        self.betas = cosine_beta_schedule(timesteps).to(self.device)
        self.num_timesteps = self.betas.shape[0]

        alphas = 1. - self.betas

        # print(self.betas.shape, self.betas.mean(), self.betas.min(), self.betas.max())
        # print(alphas.shape, alphas.mean(), alphas.min(), alphas.max())

        # TODO 3.1: compute the cumulative products for current and previous timesteps
        self.alphas_cumprod = torch.cumprod(alphas, dim =-1)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.], device=self.device),
                                              self.alphas_cumprod[:-1]])

        # TODO 3.1: pre-compute values needed for forward process
        # This is the coefficient of x_t when predicting x_0
        self.x_0_pred_coef_1 = 1./torch.sqrt(self.alphas_cumprod)
        # This is the coefficient of pred_noise when predicting x_0
        self.x_0_pred_coef_2 = -torch.sqrt(1-self.alphas_cumprod)/torch.sqrt(self.alphas_cumprod) #TODO: how do get x_o apply these at multiple timesteps rights?

        # TODO 3.1: compute the coefficients for the mean
        # This is coefficient of x_0 in the DDPM section
        self.posterior_mean_coef1 = torch.sqrt(self.alphas_cumprod_prev) * self.betas / (1-self.alphas_cumprod)
        # This is coefficient of x_t in the DDPM section
        self.posterior_mean_coef2 = torch.sqrt(alphas) * (1-self.alphas_cumprod_prev)/(1-self.alphas_cumprod)

        # TODO 3.1: compute posterior variance
        # calculations for posterior q(x_{t-1} | x_t, x_0) in DDPM

        self.posterior_variance = (1-self.alphas_cumprod_prev)/(1-self.alphas_cumprod) * self.betas
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min =1e-20))

        # sampling related parameters
        self.sampling_timesteps = default(sampling_timesteps, timesteps) # default num sampling timesteps to number of timesteps at training
        print("timesteps")
        print(self.sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

    def get_posterior_parameters(self, x_0, x_t, t):
        # TODO 3.1: Compute the posterior mean and variance for x_{t-1}
        # using the coefficients, x_t, and x_0
        # hint: can use extract function from utils.py
        
        #TODO: see how x_o is being passed -- is it the same value across multiple timesteps?
        posterior_mean = self.posterior_mean_coef1[t]*x_0 + self.posterior_mean_coef2[t]*x_t
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]


        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_t, t):
        # TODO 3.1: given a noised image x_t, predict x_0 and the additive noise
        # to predict the additive noise, use the denoising model.
        # Hint: You can use extract function from utils.py.
        # clamp x_0 to [-1, 1]

        print("FID debug, here")
        print(x_t.shape, t.shape)
        pred_noise = self.model(x_t,t)
        # print("Here")
        # print(t)
        # print(self.x_0_pred_coef_1.shape, self.x_0_pred_coef_2.shape)
        # print(self.x_0_pred_coef_1[t].shape, x_t.shape)
        # print(self.x_0_pred_coef_2[t].shape, pred_noise.shape)

        num_t = t.shape[0]
        x_0 = self.x_0_pred_coef_1[t].reshape((num_t,1,1,1)) * x_t + self.x_0_pred_coef_2[t].reshape((num_t,1,1,1)) * pred_noise
        x_0 = x_0.clamp(-1,1)

        return (pred_noise, x_0)

    @torch.no_grad()
    def predict_denoised_at_prev_timestep(self, x, t: int):
        # TODO 3.1: given x at timestep t, predict the denoised image at x_{t-1}.
        # also return the predicted starting image.
        # Hint: To do this, you will need a predicted x_0. Which function can do this for you?

        # print('here 1')
        num_t = t.shape[0]
        _, pred_x_0 = self.model_predictions(x,t)

        # print(t.shape)
        # print(pred_x_0)
        # print(self.posterior_mean_coef2[t].shape)
        # print(x.shape)

        u_t = self.posterior_mean_coef2[t].reshape((num_t,1,1,1))*x + self.posterior_mean_coef1[t].reshape((num_t,1,1,1))*pred_x_0
        sigma_t = torch.sqrt(self.posterior_variance[t])

        #TODO: we are getting variance but need Std-dev, so take square-root?
        # print("Hereeee")
        # print(u_t.shape, sigma_t.shape, x.shape)
        pred_img = u_t + sigma_t.reshape((num_t,1,1,1)) * torch.randn(x.shape).to(self.device)
        x_0 = pred_x_0

        return pred_img, x_0

    @torch.no_grad()
    def sample_ddpm(self, shape, z):
        img = z
        for t in tqdm(range(self.num_timesteps-1, 0, -1)):
            batched_times = torch.full((img.shape[0],), t, device=self.device, dtype=torch.long)
            img, _ = self.predict_denoised_at_prev_timestep(img, batched_times)
        img = unnormalize_to_zero_to_one(img)
        
        print("finale 2")
        print(img.shape, img.min(), img.max(), img.mean())
        return img

    def sample_times(self, total_timesteps, sampling_timesteps):
        # TODO 3.2: Generate a list of times to sample from.
        times = torch.arange(total_timesteps,0, -sampling_timesteps).to(self.device)-1
        print("In sample times")
        print(times)
        return times

    def get_time_pairs(self, times):
        # TODO 3.2: Generate a list of adjacent time pairs to sample from.
        return torch.stack((times[:-1], times[1:]), dim=1)
        #TODO: Rev it

    def ddim_step(self, batch, device, tau_i, tau_isub1, img, model_predictions, alphas_cumprod, eta):
        # TODO 3.2: Compute the output image for a single step of the DDIM sampling process.
        print("in ddim step")
        print(tau_i)
        print(img.shape)

        # predict x_0 and the additive noise for tau_i
        pred_noise, x_0 = model_predictions(img, tau_i.repeat(batch))

        # extract \alpha_{\tau_{i - 1}} and \alpha_{\tau_{i}}
        alpha_tau_i = self.alphas_cumprod[tau_i]
        alpha_tau_isub1 = self.alphas_cumprod[tau_isub1]

        # compute \sigma_{\tau_{i}}
        var_tau_i = eta * (1-alpha_tau_isub1)/(1-alpha_tau_i)*self.betas[tau_isub1]
        sigma_tau_i = torch.sqrt(var_tau_i)

        # compute the coefficient of \epsilon_{\tau_{i}}
        
        print("ddim step 2", alpha_tau_isub1.shape)
        u_tau_i = torch.sqrt(alpha_tau_isub1)*x_0 + torch.sqrt(1-alpha_tau_isub1-var_tau_i)*pred_noise

        print(var_tau_i.shape, sigma_tau_i.shape, u_tau_i.shape)
        print(alpha_tau_i.shape, alpha_tau_isub1.shape)

        # sample from q(x_{\tau_{i - 1}} | x_{\tau_t}, x_0)
        # HINT: use the reparameterization trick
        #TODO: deterministic or stochastic here sampling?
        img = u_tau_i + sigma_tau_i*torch.randn(img.shape).to(self.device)

        print(img.shape)

        return img, x_0

    def sample_ddim(self, shape, z):
        #TODO: sampling timesteps theek hai? just 10 timesteps? and that tau_0=100 and not tau_0=0

        print("here ddim")
        print(shape, z.shape)

        batch, device, total_timesteps, sampling_timesteps, eta = shape[0], self.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta

        times = self.sample_times(total_timesteps, sampling_timesteps)
        time_pairs = self.get_time_pairs(times)

        print("here 3")
        print(total_timesteps, sampling_timesteps)
        print(times.shape, time_pairs.shape)
        print(times)
        print(time_pairs)

        img = z
        print("finale 1", img.shape)
        for tau_i, tau_isub1 in tqdm(time_pairs, desc='sampling loop time step'):
            print("inside loop", tau_i, tau_isub1, batch)
            img, _ = self.ddim_step(batch, device, tau_i, tau_isub1, img, self.model_predictions, self.alphas_cumprod, eta)

        img = unnormalize_to_zero_to_one(img)
        print("finale 2")
        print(img.shape, img.min(), img.max(), img.mean())
        return img

    @torch.no_grad()
    def sample(self, shape):
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        print("in sample")
        print(shape)
        z = torch.randn(shape, device = self.betas.device)
        return sample_fn(shape, z)

    @torch.no_grad()
    def sample_given_z(self, z, shape):
        #TODO 3.3: fill out based on the sample function above
        sample_fn = self.sample_ddpm if not self.is_ddim_sampling else self.sample_ddim
        return sample_fn(shape, z)
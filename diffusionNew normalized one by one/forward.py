import torch.nn.functional as F
import torch
import numpy as np

def linear_beta_scheadule(timesteps, start = 0.0001 , end=0.07):
    """
    Parameters:
    timesteps (int): The number of timesteps for which to generate the schedule.
    start (float, optional): The starting value of beta. Default is 0.0001.
    end (float, optional): The ending value of beta. Default is 0.02.

    Returns:
    torch.Tensor: A 1D tensor of length 'timestep' with values linearly spaced from 'start' to 'end'.
    """
    return torch.linspace(start, end, timesteps)

def qudratic_beta_scheadule(timesteps, start=1*10e-6, end=0.5):
    """
    Parameters:
    timesteps (int): The number of timesteps for which to generate the schedule.
    start (float, optional): The starting value of beta. Default is 0.0001.
    end (float, optional): The ending value of beta. Default is 0.5.

    Returns:
    torch.Tensor: A 1D tensor of length 'timestep' with values quadratically spaced from 'start' to 'end'.
    beta_t = (sqrt(beta_0) + t*(sqrt(beta_T)-sqrt(beta_0))/T)^2
    """
    betas = torch.zeros(timesteps)
    for t in range(timesteps):
        betas[t] = (np.sqrt(start) + t * (np.sqrt(end) - np.sqrt(start)) / timesteps) ** 2
    return betas


def get_index_from_list(list, t, x_shape):
    batch_size = t.shape[0]
    out = list.gather(-1,t.cpu())
    return out.reshape(batch_size, *( (1,) * (len(x_shape)-1) )).to(t.device)

def forward_diffusion_sample(x_0,t,device="cpu"):
    # Generate noise with the same shape as x_0
    noise = torch.randn_like(x_0)

    # Get the cumulative product of sqrt(alpha) at time t
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)

    # Get the cumulative product of sqrt(1-alpha) at time t
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    # Compute the forward diffusion process
    # The new state is a combination of the original state scaled by sqrt_alphas_cumprod_t and the noise scaled by sqrt_one_minus_alphas_cumprod_t
    return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)

# We predefine the parameters
T = 50
# betas = linear_beta_scheadule(T)
betas = qudratic_beta_scheadule(T)

alphas = 1.-betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1. / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas*(1-alphas_cumprod_prev)/(1-alphas_cumprod)



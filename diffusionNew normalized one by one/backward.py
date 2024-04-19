import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
from tqdm import tqdm
import torch.optim as optim
import math
import forward
from torch.optim import Adam
import matplotlib.pyplot as plt
from transformers import InformerConfig, InformerModel


# write the parameters values
T = forward.T
betas = forward.linear_beta_scheadule(T)

def show_tensor_TS(x):
    x = x[0]  # Select the first time series in the batch
    plt.plot(x)  # Display the time series



class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class ActivationGate(nn.Module):
    def __init__(self):
        super(ActivationGate, self).__init__()

    def forward(self, x):
        return torch.sigmoid(x)

# Define the architecture of the denoising network
class DenoisingNetwork(nn.Module):
    def __init__(self, input_size=1010, channel_size = 64,  hidden_size1=50, hidden_size2=10, hidden_size3=50, output_size=100):
        super(DenoisingNetwork, self).__init__()

        self.conv1 = nn.Conv1d(1, channel_size, kernel_size=3, padding=1)

        self.temporal = PositionalEncoding(d_model=128, dropout=0.1, max_len=5000)

        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(hidden_size2, hidden_size3)
        self.dense3 = nn.Linear(hidden_size3, hidden_size1)
        self.dropout = nn.Dropout(0.5)
<<<<<<< HEAD
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=100, nhead=4), num_layers=2, enable_nested_tensor=False)
        
=======

        configuration = InformerConfig(prediction_length=100)
        self.transformer = InformerModel(configuration)

>>>>>>> abb909be45e06c194c2689f714951020dd04a6a2
        self.conv2 = nn.Conv1d(64, 64 , kernel_size=3, padding=1)
        
        self.activation_gate = ActivationGate()

        self.conv3 = nn.Conv1d(64 , 64 , kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(64 , 1 , kernel_size=3, padding=1)

    def forward(self, x, t):
        # Process the time series
        x = x.float().unsqueeze(1)
        x_res = F.relu(self.conv1(x))
        x = self.dropout(x_res)

        # Process the time step
        t = self.temporal.pe[t, 0, :]
        t = F.silu(self.dense1(t))
        
        # Unsqueeze t to make its shape (batch_size, 1, d_model)
        t = t.unsqueeze(-1)
        concat = x + t
        
        # Process the concatenated output
        out = self.transformer(concat)
        out = self.conv2(out)
        out = self.activation_gate(out)
        out = self.conv5(out)
        return out.squeeze()
    
def reverse_diffusion_process(noisy_data, t, denoising_network):
    return denoising_network(noisy_data, t)

def get_loss(model, x_0, t):
    x_noisy, noise = forward.forward_diffusion_sample(x_0, t)
    noise_pred = model(x_noisy, t)
    return F.l1_loss(noise,noise_pred)

@torch.no_grad()
def sample_timestep(x,t,model):
    betas_t = forward.get_index_from_list(forward.betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = forward.get_index_from_list(forward.sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = forward.get_index_from_list(forward.sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance = forward.get_index_from_list(forward.posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance) * noise

device = "cuda" if torch.cuda.is_available() else "cpu"

@torch.no_grad()
def sample_plot_TS(model):
    x = torch.randn(1, 100)
    
    # Prepare the plot
    plt.figure(figsize=(100, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(T/num_images)
    
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i ,device = device, dtype=torch.long)
        x = sample_timestep(x, t, model)

        if i % stepsize == 0:
            plt.subplot(1, num_images, i//stepsize + 1)
            show_tensor_TS(x.detach().cpu())
    plt.show()

def sample_TS(model):
    x = torch.randn(1, 100)
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i ,device = device, dtype=torch.long)
        x = sample_timestep(x, t, model)
    return x


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset

# import the normalized slices
slices = pd.read_csv('data/slices_normalized.csv', index_col=0)

# Convert the DataFrame to a numpy array, then to a PyTorch Tensor
slices_tensor = torch.tensor(slices.values)

# Convert your data to a TensorDataset
slices_dataset = TensorDataset(slices_tensor)

# DEFINES THE BATCH SIZE
batch_size = 32

# Create a DataLoader with shuffle=True for shuffling at each epoch
slices_loader = DataLoader(slices_dataset, batch_size=batch_size, shuffle=True, drop_last=True)


# import the backward file and functions
import backward
from torch.optim import Adam

# Create an instance of the denoising network
model = backward.DenoisingNetwork()


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
optimizer = Adam(model.parameters(), lr=0.00001)
epochs = 100
T = backward.T

losses = []
# Train the model
print("Training the model...")
for epoch in tqdm(range(epochs)):
    for step, batch in enumerate(slices_loader):
        optimizer.zero_grad()

        t = torch.randint(0,T, (batch_size,), device=device).long()
        
        loss = backward.get_loss(model, batch[0], t) 
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0 and step == 0:
            print(f"Epoch {epoch} | step {step:03d}, Loss {loss.item()}")
            losses.append((epoch, loss.item()))
            backward.sample_plot_TS(model)        

def sample_TS(model):
    x = torch.randn(1, 100)
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i ,device = device, dtype=torch.long)
        x = backward.sample_timestep(x, t, model)
    return x

# use the model to generate 1000 samples
samples = []
for i in tqdm(range(1000)):
    TS = sample_TS(model)
    # Normalize the tensor between -1 and 1
    TS_norm = 2 * (TS - TS.min()) / (TS.max() - TS.min()) - 1
    # Convert tensor to numpy array and flatten it to 1D
    samples.append(TS_norm.squeeze(0).cpu().numpy())

# Convert the samples to a DataFrame
samples_df = pd.DataFrame(samples)

# Save the samples to a CSV file without index
samples_df.to_csv('data/generated_samples.csv', index=False)
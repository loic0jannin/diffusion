import pandas as pd
import numpy as np
import os
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import TensorDataset

# import the normalized slices
slices = pd.read_csv('data/slices_normalized.csv')

# Convert the DataFrame to a numpy array, then to a PyTorch Tensor
slices_tensor = torch.tensor(slices.values)

# Convert your data to a TensorDataset
slices_dataset = TensorDataset(slices_tensor)

# DEFINES THE BATCH SIZE
batch_size = 32

# Create a DataLoader with shuffle=True for shuffling at each epoch
slices_loader = DataLoader(slices_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
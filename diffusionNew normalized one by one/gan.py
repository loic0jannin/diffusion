import torch
import torch.nn as nn
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.autograd.variable import Variable
import torch.nn.functional as F
torch.autograd.set_detect_anomaly(True)
from torch.utils.data import DataLoader, TensorDataset

# import the normalized slices
slices = pd.read_csv('data/slices_normalized.csv', index_col=0)

# Convert the DataFrame to a numpy array, then to a PyTorch Tensor
slices_tensor = torch.tensor(slices.values)

# Convert your data to a TensorDataset
slices_dataset = TensorDataset(slices_tensor)

# DEFINES THE BATCH SIZE
batch_size = 32

# Create a DataLoader with shuffle=True for shuffling at each epoch
train_loader = DataLoader(slices_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

N = 100

# Define the Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(N, 2*N), # Input is a N slicing window
            nn.ReLU(),
            nn.Linear(2*N,2*N), 
            nn.ReLU(),
            nn.Linear(2*N, N),  # Output is N slicing window
        )

    def forward(self, x):
         x = x.double()
         output = self.model(x)
         return output

# Define the Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(N, 2*N),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(2*N, N),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(N, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
         output = self.model(x)
         return output


# Create the Generator and Discriminator
generator = Generator()
discriminator = Discriminator()

# Define the loss function and optimizers
criterion = nn.BCELoss()
criterion_autoencoder = nn.MSELoss()  # Separate loss function for the autoencoder
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.00005)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=0.00005)


# Initialize the CSV file
with open('losses.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Epoch", "Discriminator Loss", "Generator Loss"])


print("Training the GAN...")
# Train the GAN
for epoch in tqdm(range(500)):
    for index, (real_data,) in enumerate(train_loader):

        
        # Preparing the real data to train the discriminator:
        real_data_label = torch.ones(batch_size,1)

        # Preparing the fake data to train the discriminator: 
        noise_data_set = torch.randn((batch_size, N))
        fake_data_set = generator(noise_data_set)
        fake_data_label = torch.zeros(batch_size, 1)


        # Creating the training samples set:
        training_data_set = torch.cat((real_data, fake_data_set))

        # Creating the training labels set:
        training_labels_set = torch.cat((real_data_label, fake_data_label))

        # Train the discriminator:
        discriminator.zero_grad()
        output_discriminator = discriminator(training_data_set)
        loss_discriminator = criterion(output_discriminator, training_labels_set)
        loss_discriminator.backward()
        optimizer_discriminator.step()

        # Initialising the data for the generator: 
        noise_data_set = torch.randn((batch_size, N))

        # Train the generator:  
        generator.zero_grad()
        output_generator = generator(noise_data_set)
        output_discriminator_generated = discriminator(output_generator)
        loss_generator = criterion(output_discriminator_generated, real_data_label)
        loss_generator.backward()
        optimizer_generator.step()

    # Write the progress to the CSV file
    if epoch % 10 == 0:
        with open('losses.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([epoch, loss_discriminator.item(), loss_generator.item()])



def get_random_sample_from_generator(generator, N, batch_size,num_samples):
    samples = []

    for _ in range(num_samples):
        # Generate random noise
        noise = torch.randn((batch_size, N))

        # Generate samples from the GAN
        with torch.no_grad():
            sample = generator(noise)

        # Convert the first sample to a numpy array
        sample = sample[0].numpy()

        # Invert the scaling
        samples.append(sample)

    return np.array(samples)


# save 1000 generated samples to a CSV file
generated_samples = get_random_sample_from_generator(generator, N, batch_size, 1000)

# Save the generated samples to a CSV file
np.savetxt('TS/generated_samples.csv', generated_samples, delimiter=',')

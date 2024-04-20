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

# define the lstm model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, 100)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (c0, h0))
        out = self.fc(out[:, -1, :])
        return out


class Generator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.2):
        super(Generator, self).__init__()
        
        # Create an instance of LSTMModel
        self.lstm = LSTMModel(input_size, hidden_size, num_layers, dropout_rate)
        # Convolutional layer from 1 channel to 64
        self.conv1 = nn.Conv1d(1, 64, 1)
        # Convolutional layer from 64 channels back to 1
        self.conv2 = nn.Conv1d(64, 1, 1)

    def forward(self, x):
        # Pass the input through the LSTMModel
        output = self.lstm(x)
        # Add an extra dimension for the number of channels
        output = output.unsqueeze(1)  # Now the shape is (batch_size, 1, seq_len)
        first_conv = self.conv1(output)
        second_conv = self.conv2(first_conv)
        # Remove the channel dimension
        output = second_conv.squeeze(1)  
        return output



class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout_rate=0.2):
        super(Discriminator, self).__init__()
        
        # Create an instance of LSTMModel
        self.lstm = LSTMModel(input_size, hidden_size, num_layers, dropout_rate)
        # Convolutional layer from 1 channel to 64
        self.conv1 = nn.Conv1d(1, 64, 1)
        # Convolutional layer from 64 channels back to 1
        self.conv2 = nn.Conv1d(64, 1, 1)
        # funlly connected layer that match the output of the conv 2 
        self.fc = nn.Linear(100, 1)
        # sigmoid layer
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Pass the input through the LSTMModel
        output = self.lstm(x)
        # Add an extra dimension for the number of channels
        output = output.unsqueeze(1)  # Now the shape is (batch_size, 1, seq_len)
        first_conv = self.conv1(output)
        second_conv = self.conv2(first_conv)
        output = self.fc(second_conv)
        # Remove the channel dimension
        output = output.squeeze(1)
        output = self.sigmoid(output)
        return output


# Create the Generator and Discriminator
generator = Generator( input_size = 1, hidden_size = 256 , num_layers = 1)
discriminator = Discriminator( input_size = 1, hidden_size = 256 , num_layers = 1)

# Define the loss function and optimizers
criterion = nn.BCELoss()
criterion_autoencoder = nn.MSELoss()  # Separate loss function for the autoencoder
optimizer_discriminator = torch.optim.Adam(discriminator.parameters(), lr=0.0005)
optimizer_generator = torch.optim.Adam(generator.parameters(), lr=0.0005)


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
        training_data_set = torch.cat((real_data, fake_data_set)).float()

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
            print(f"Epoch {epoch}, Discriminator Loss: {loss_discriminator.item()}, Generator Loss: {loss_generator.item()}",flush=True)



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

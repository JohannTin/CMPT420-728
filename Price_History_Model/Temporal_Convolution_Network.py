import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import os  # Import os to check for file existence
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import weight_norm

import matplotlib.pyplot as plt

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.W = nn.Linear(input_dim, input_dim, bias=False)
        self.V = nn.Linear(input_dim, 1, bias=False)

    def forward(self, x):
        # x shape: (batch_size, num_channels, seq_length)
        batch_size, channels, seq_length = x.size()
        x = x.permute(0, 2, 1)  # Change to (batch_size, seq_length, num_channels)
        attention_weights = torch.tanh(self.W(x))  # (batch_size, seq_length, num_channels)
        attention_weights = self.V(attention_weights)  # (batch_size, seq_length, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)  # (batch_size, seq_length, 1)
        attention_weights = attention_weights.permute(0, 2, 1)  # (batch_size, 1, seq_length)

        # Compute the context vector
        context_vector = torch.bmm(attention_weights, x)  # (batch_size, 1, num_channels)
        return context_vector.squeeze(1)  # (batch_size, num_channels)

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.attention = Attention(num_channels[-1])  # Initialize attention layer

    def forward(self, x):
        x = self.network(x)
        x = self.attention(x.unsqueeze(1))  # Add a dummy dimension for seq_length
        return x

# Load stock data (without sentiment scores atm)
df = pd.read_csv("aapl_data.csv", parse_dates=["Date"], index_col="Date")

# Select Dates from 2019 Jan 1 - 2025 Jan 1
start_date = '2024-12-01'
end_date = '2025-01-01'

filtered_df = df.loc[start_date:end_date]

# Select features: closing price and sentiment
data = filtered_df[["Open", "High", "Low", "Close", "Volume"]].values  
# data = df[["Close", "Sentiment"]].values  

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)



# Define parameters
seq_length = 10  # Define your desired sequence length
num_features = data.shape[1]  # Should be 5 for Open, High, Low, Close, Volume

# Create sequences from the data
sequences = []
for i in range(len(data) - seq_length):
    seq = data[i:i + seq_length]  # Create a sequence of length `seq_length`
    sequences.append(seq)

# Convert to numpy array and transpose to fit (batch_size, num_channels, seq_length)
sequences = np.array(sequences)
sequences = sequences.transpose(0, 2, 1)  # Shape: (batch_size, num_channels, seq_length)

class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32)

# Create dataset and dataloader
dataset = TimeSeriesDataset(sequences)
dataloader = DataLoader(dataset, batch_size=32)  # Adjust batch size as needed

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

# # Convert to PyTorch tensors
# X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
# X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# # Create DataLoader
# train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
# train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# # Initialize model, loss function, and optimizer
# model_path = 'tcn_with_attention_model.pth'
# model = TCNWithAttention(num_features=5, num_channels=64)  # 2 features: price and weighted sentiment

# # Check if model file exists
# if os.path.exists(model_path):
#     # Load the model
#     model.load_state_dict(torch.load(model_path))
#     model.eval()  # Set to evaluation mode
#     print("Model loaded from", model_path)
# else:
#     # Assuming your model is defined as `model`
model = TemporalConvNet(num_inputs=num_features, num_channels=[16, 32, 64], kernel_size=2, dropout=0.2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # Adjust learning rate as needed
criterion = nn.MSELoss()  # Adjust loss function based on your task

# Training loop
model.train()
num_epochs = 100  # Define the number of epochs
for epoch in range(num_epochs):
    for batch in dataloader:
        optimizer.zero_grad()  # Clear previous gradients
        outputs = model(batch)  # Forward pass
        loss = criterion(outputs, batch[:, :, -1])  # Compute loss; adjust based on your target
        loss.backward()  # Backward pass
        optimizer.step()  # Update weights


# Generate predictions
model.eval()  # Set the model to evaluation mode
with torch.no_grad():  # No need to track gradients
    predictions = model(torch.tensor(sequences, dtype=torch.float32)).numpy()  # Forward pass

# Assuming 'targets' are the true values we want to compare with (e.g., next Close price)
# You may need to adjust how you extract your targets based on your specific use case
# For simplicity, let's say we take the last feature of the last time step as the target
targets = data[seq_length:, -1]  # True values

# Plotting
plt.figure(figsize=(14, 7))
plt.plot(targets, label='True Values', color='blue', alpha=0.7)
plt.plot(predictions[:, -1], label='Predictions', color='red', alpha=0.7)  # Assuming predictions are of shape (batch_size, num_channels)
plt.title('Predictions vs. True Values')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()

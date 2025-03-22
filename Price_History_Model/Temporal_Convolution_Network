import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import os  # Import os to check for file existence

# Define Attention Layer
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(input_dim, input_dim)
        self.V = nn.Linear(input_dim, 1)

    def forward(self, x):
        score = self.V(torch.tanh(self.W(x)))  # Compute attention scores
        attention_weights = F.softmax(score, dim=1)  # Softmax to get attention weights
        context_vector = attention_weights * x  # Weighted sum of inputs
        return context_vector, attention_weights

# Define TCN with Attention and Confidence Interval Output
class TCNWithAttention(nn.Module):
    def __init__(self, num_features, num_channels):
        super(TCNWithAttention, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(num_features, num_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.attention = AttentionLayer(num_channels)  # Attention layer after TCN
        self.fc_price = nn.Linear(num_channels, 3)  # Output for next 3 days (mean)
        self.fc_uncertainty = nn.Linear(num_channels, 3)  # Output for uncertainty (std dev)

    def forward(self, x):
        # x: [batch_size, seq_len, num_features] -> [batch_size, num_features, seq_len]
        x = x.permute(0, 2, 1)
        x = self.tcn(x)  # Apply TCN layers
        x = x.permute(0, 2, 1)  # [batch_size, seq_len, num_channels]
        
        # Apply attention mechanism
        context_vector, attention_weights = self.attention(x)
        x_mean = self.fc_price(context_vector.mean(dim=1))  # Mean prediction
        x_std = F.softplus(self.fc_uncertainty(context_vector.mean(dim=1)))  # Std dev prediction
        return x_mean, x_std, attention_weights

# Function to create sequences from data
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len - 3):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+3, 0])  # Predict the closing price
    return np.array(X), np.array(y)

# Load stock data (with sentiment scores)
df = pd.read_csv("stock_prices_with_sentiment.csv", parse_dates=["Date"], index_col="Date")

# Select features: closing price and sentiment
data = df[["Close", "Sentiment"]].values  

# Normalize data
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Create sequences
seq_len = 10
X, y = create_sequences(data_scaled, seq_len)

# Apply weights based on the magnitude of sentiment
weights = np.abs(X[:, :, 1])  # Get absolute sentiment values
X_weighted = X.copy()
X_weighted[:, :, 1] = X_weighted[:, :, 1] * weights  # Weight the sentiment feature

# Split into training and testing sets
train_size = int(len(X_weighted) * 0.8)
X_train, X_test = X_weighted[:train_size], X_weighted[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

# Create DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Initialize model, loss function, and optimizer
model_path = 'tcn_with_attention_model.pth'
model = TCNWithAttention(num_features=2, num_channels=64)  # 2 features: price and weighted sentiment

# Check if model file exists
if os.path.exists(model_path):
    # Load the model
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set to evaluation mode
    print("Model loaded from", model_path)
else:
    # Initialize optimizer and loss function if creating a new model
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    for epoch in range(50):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            mean_predictions, std_predictions, _ = model(X_batch)  # Forward pass
            loss = criterion(mean_predictions, y_batch)  # Compute loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
        print(f'Epoch {epoch+1}, Loss: {loss.item():.4f}')
    
    # Save the model
    torch.save(model.state_dict(), model_path)
    print("Model saved to", model_path)

# Make predictions on test data
with torch.no_grad():
    mean_pred, std_pred, attention_weights = model(X_test_tensor)

# Display predictions
predicted_prices = mean_pred.numpy()
predicted_stds = std_pred.numpy()

# Calculate confidence intervals
confidence_intervals = 1.96 * predicted_stds  # 95% confidence intervals
lower_bounds = predicted_prices - confidence_intervals
upper_bounds = predicted_prices + confidence_intervals

# Display results for the last predicted day
for i in range(3):
    print(f"Day {i+1}: Predicted Price: {predicted_prices[-1][i]:.4f}, "
          f"Confidence Interval: [{lower_bounds[-1][i]:.4f}, {upper_bounds[-1][i]:.4f}]")

# Visualization (Optional)
import matplotlib.pyplot as plt

# Plot historical and predicted values
plt.figure(figsize=(12, 6))
plt.plot(df.index[-len(y_test):], y_test, label="True Prices", color='blue')
plt.plot(df.index[-len(y_test):], predicted_prices, label="Predicted Prices", color='orange')
plt.fill_between(df.index[-len(y_test):],
                 lower_bounds.flatten(), upper_bounds.flatten(), color='lightgrey', alpha=0.5,
                 label="95% Confidence Interval")
plt.title("True vs Predicted Prices with Confidence Intervals")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.show()

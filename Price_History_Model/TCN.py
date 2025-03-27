import torch.nn as nn
import numpy as np
import torch
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Load stock data
df = pd.read_csv("aapl_data.csv", parse_dates=["Date"], index_col="Date")

# Check if MPS is available, otherwise fallback to CPU or CUDA
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

# Select Dates from 2020 Dec 1 - 2025 Jan 1
start_date = '2020-12-01'
end_date = '2025-01-01'

filtered_df = df.loc[start_date:end_date]

# Feature Engineering - Add technical indicators
def add_technical_indicators(df):
    # Copy the dataframe to avoid modifying the original
    df_with_features = df.copy()
    
    # Moving averages
    df_with_features['MA5'] = df_with_features['Close'].rolling(window=5).mean()
    df_with_features['MA10'] = df_with_features['Close'].rolling(window=10).mean()
    df_with_features['MA20'] = df_with_features['Close'].rolling(window=20).mean()
    
    # Relative Strength Index (RSI)
    delta = df_with_features['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df_with_features['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df_with_features['BB_middle'] = df_with_features['Close'].rolling(window=20).mean()
    std = df_with_features['Close'].rolling(window=20).std()
    df_with_features['BB_upper'] = df_with_features['BB_middle'] + 2 * std
    df_with_features['BB_lower'] = df_with_features['BB_middle'] - 2 * std
    
    # MACD (Moving Average Convergence Divergence)
    exp1 = df_with_features['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df_with_features['Close'].ewm(span=26, adjust=False).mean()
    df_with_features['MACD'] = exp1 - exp2
    df_with_features['MACD_signal'] = df_with_features['MACD'].ewm(span=9, adjust=False).mean()
    
    # Percentage price change
    df_with_features['Price_Change'] = df_with_features['Close'].pct_change()
    
    # Volatility (standard deviation over a window)
    df_with_features['Volatility'] = df_with_features['Close'].pct_change().rolling(window=10).std()
    
    # Drop NaN values created by the indicators
    df_with_features = df_with_features.dropna()
    
    return df_with_features

# Apply feature engineering
enhanced_df = add_technical_indicators(filtered_df)

# Prepare your data
# Selecting the most important features
feature_columns = [
    "Open", "High", "Low", "Close", "Volume",
    "MA5", "MA10", "MA20", "RSI", "MACD", "MACD_signal",
    "BB_upper", "BB_lower", "Price_Change", "Volatility"
]

data = enhanced_df[feature_columns].values

# Normalize the data
data_mean = data.mean(axis=0)
data_std = data.std(axis=0)
data_normalized = (data - data_mean) / data_std

# Create sequences and labels for next day prediction
def create_sequences(data, seq_length, predict_ahead=1):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length - predict_ahead + 1):
        seq = data[i:i + seq_length]
        # Predict the next day's Open price
        label = data[i + seq_length + predict_ahead - 1, 0]  
        sequences.append(seq)
        labels.append(label)
    return np.array(sequences), np.array(labels)

# Prepare sequence data
seq_length = 35  # Increased sequence length to capture more patterns
predict_ahead = 1  # Predict 1 day ahead
X, y = create_sequences(data_normalized, seq_length, predict_ahead)

# Split into training and validation sets (80/20 split)
split_idx = int(len(X) * 0.8)
X_train, X_val = X[:split_idx], X[split_idx:]
y_train, y_val = y[:split_idx], y[split_idx:]

# Save original y values for later comparison
y_val_original = y_val.copy()
y_train_original = y_train.copy()

# Convert to PyTorch tensors and move to device
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

# Permute X tensors for convolution
X_train_tensor = X_train_tensor.permute(0, 2, 1)
X_val_tensor = X_val_tensor.permute(0, 2, 1)

print(f"X_train_tensor shape: {X_train_tensor.shape}")
print(f"y_train_tensor shape: {y_train_tensor.shape}")

# Enhanced TCN model with Attention
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.query = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv1d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv1d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Query, Key, Value projections
        proj_query = self.query(x).permute(0, 2, 1)  # B x L x C//8
        proj_key = self.key(x)  # B x C//8 x L
        energy = torch.bmm(proj_query, proj_key)  # B x L x L
        attention = F.softmax(energy, dim=2)  # B x L x L
        
        proj_value = self.value(x)  # B x C x L
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))  # B x C x L
        
        out = self.gamma * out + x
        return out

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(0.1)  # Using LeakyReLU instead of ReLU
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU(0.1)
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight, nonlinearity='leaky_relu')
        nn.init.kaiming_normal_(self.conv2.weight, nonlinearity='leaky_relu')
        if self.downsample is not None:
            nn.init.kaiming_normal_(self.downsample.weight, nonlinearity='leaky_relu')

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class ImprovedTemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.3):
        super(ImprovedTemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.attention = SelfAttention(num_channels[-1])
        
        # Multiple fully connected layers with batch normalization
        self.fc_layers = nn.Sequential(
            nn.BatchNorm1d(num_channels[-1]),
            nn.Linear(num_channels[-1], 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.BatchNorm1d(64),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Input shape: [batch, features, sequence]
        features = self.network(x)
        # Apply attention mechanism
        features = self.attention(features)
        # Take features from last time step
        last_features = features[:, :, -1]
        # Apply fully connected layers
        return self.fc_layers(last_features).squeeze(-1)

# Create an instance of the improved model
num_inputs = X_train_tensor.shape[1]  # Number of features
num_channels = [64, 128, 256, 128]  # Deeper network with more channels
model = ImprovedTemporalConvNet(num_inputs, num_channels, kernel_size=3, dropout=0.3)
model = model.to(device)

# Define loss function and optimizer with weight decay
criterion = nn.HuberLoss(delta=1.0)  # Huber loss is more robust to outliers than MSE
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

# Early stopping implementation
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# Initialize early stopping
early_stopping = EarlyStopping(patience=70)

# Training loop with early stopping
num_epochs = 200  # More epochs with early stopping
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    
    # Forward pass
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    
    # Backpropagation
    loss.backward()
    
    # Gradient clipping to prevent exploding gradients
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    optimizer.step()
    
    # Validation
    model.eval()
    with torch.no_grad():
        val_output = model(X_val_tensor)
        val_loss = criterion(val_output, y_val_tensor)
        val_losses.append(val_loss.item())
    
    # Store the loss
    train_losses.append(loss.item())
    
    # Update learning rate scheduler
    scheduler.step(val_loss)
    
    # Check if this is the best model so far
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict().copy()
    
    # Early stopping check
    early_stopping(val_loss.item())
    if early_stopping.early_stop:
        print(f"Early stopping triggered at epoch {epoch+1}")
        break
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

# Load the best model
if best_model_state is not None:
    model.load_state_dict(best_model_state)

# Make predictions with the best model
model.eval()
with torch.no_grad():
    test_predictions = model(X_val_tensor)
    test_predictions = test_predictions.cpu().numpy()

# Denormalize predictions
test_predictions_denormalized = test_predictions * data_std[0] + data_mean[0]

# Denormalize truth values
y_val_denormalized = y_val_original * data_std[0] + data_mean[0]

# Get the corresponding dates for validation set
val_dates = enhanced_df.index[split_idx + seq_length:split_idx + seq_length + len(y_val)]

# Plot the predictions against the true values
plt.figure(figsize=(14, 7))
plt.plot(val_dates, y_val_denormalized, label="True Open Prices", color="blue", marker='o', markersize=3, linestyle="-")
plt.plot(val_dates, test_predictions_denormalized, label="Predicted Open Prices", color="red", marker='x', markersize=3, linestyle="--")
plt.title("Apple Stock: True vs. Predicted Open Prices (Improved Model)")
plt.xlabel("Date")
plt.ylabel("Open Price ($)")
plt.grid(True, alpha=0.3)

# Format x-axis to show dates properly
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
plt.xticks(rotation=45)

# Add legend with improved formatting
plt.legend(loc='best', frameon=True, fancybox=True, shadow=True)

# Calculate and display model metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
mae = mean_absolute_error(y_val_denormalized, test_predictions_denormalized)
rmse = np.sqrt(mean_squared_error(y_val_denormalized, test_predictions_denormalized))
r2 = r2_score(y_val_denormalized, test_predictions_denormalized)

# Add metrics as text on the plot
plt.figtext(0.15, 0.05, f'MAE: ${mae:.2f}   RMSE: ${rmse:.2f}   R²: {r2:.4f}', 
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

plt.tight_layout()
plt.savefig('improved_stock_prediction.png')
plt.show()

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss (Huber)")
plt.title("Training and Validation Loss Over Time")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('improved_loss_curves.png')
plt.show()
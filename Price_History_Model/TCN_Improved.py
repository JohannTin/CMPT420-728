import torch.nn as nn
import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.models as models

CONFIG = {
    'SYMBOL': 'AAPL',
    'SEQUENCE_LENGTH': 30,
    'TRAIN_SIZE_RATIO': 0.85,
    'EPOCHS': 50,  # Increased epochs
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 0.001,
    'DROPOUT_RATE': 0.5,
    'CONFIDENCE_THRESHOLD': 0.5,  # Lowered threshold
    'START_DATE': '2010-01-01',
    'END_DATE': '2025-01-01'
}

def load_and_prepare_data():
    """Load and prepare both price and sentiment data."""
    # Load price data with indicators
    print("Loading data from appl_data_with_indicators.csv...")
    df_price = pd.read_csv('appl_data_with_indicators.csv')
    df_price['Date'] = pd.to_datetime(df_price['Unnamed: 0'])
    df_price.set_index('Date', inplace=True)
    df_price = df_price.sort_index()
    
    # Load sentiment data
    print("Loading data from sentiment_analysis_detailed.csv...")
    df_sentiment = pd.read_csv('sentiment_analysis_detailed.csv')
    # Convert published_at to datetime and extract date more robustly
    df_sentiment['Date'] = pd.to_datetime(df_sentiment['published_at'], utc=True).dt.strftime('%Y-%m-%d')
    df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date'])
    df_sentiment = df_sentiment.groupby('Date').agg({
        'sentiment_positive': 'mean',
        'sentiment_neutral': 'mean',
        'sentiment_negative': 'mean'
    }).reset_index()
    df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date'])
    df_sentiment.set_index('Date', inplace=True)
    
    # Merge price and sentiment data
    df = df_price.join(df_sentiment, how='left')
    
    # Forward fill sentiment scores for days without news
    df[['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']] = \
        df[['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']].ffill()
    
    # If there are still NaN values at the start, backward fill
    df[['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']] = \
        df[['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']].bfill()
    
    # Fill any remaining NaN values with neutral sentiment
    df[['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']] = \
        df[['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']].fillna(1/3)
    
    # Filter date range
    df = df[(df.index >= CONFIG['START_DATE']) & (df.index <= CONFIG['END_DATE'])]
    
    return df

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

def calculate_confidence(predictions, actuals):
    """Calculate confidence based on prediction error."""
    errors = np.abs(predictions - actuals)
    max_error = np.max(actuals) - np.min(actuals)
    print("max_error:", max_error)
    confidence = 1 - (errors / max_error)
    return confidence

def generate_trading_signals(predictions, actuals, confidence_threshold):
    """Generate trading signals based on predictions and confidence."""
    confidence = calculate_confidence(predictions.flatten(), actuals)
    signals = np.zeros(len(predictions))
    
    # Generate signals where confidence is high enough
    high_confidence = confidence >= confidence_threshold
    price_diff = predictions.flatten() - actuals.flatten()
    
    signals[high_confidence & (price_diff > 0)] = 1  # Buy signals
    signals[high_confidence & (price_diff < 0)] = -1  # Sell signals
    
    return signals, confidence

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

def trainModel(device, X_train_tensor, X_val_tensor, y_train_tensor, y_val_tensor):

    # Create an instance of the improved model
    num_inputs = X_train_tensor.shape[1]  # Number of features
    num_channels = [64, 128, 256, 128]  # Deeper network with more channels
    model = ImprovedTemporalConvNet(num_inputs, num_channels, kernel_size=3, dropout=CONFIG["DROPOUT_RATE"])
    model = model.to(device)

    # Define loss function and optimizer with weight decay
    criterion = nn.HuberLoss(delta=1.0)  # Huber loss is more robust to outliers than MSE
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=70)

    # Training loop with early stopping
    num_epochs = CONFIG["EPOCHS"]  # More epochs with early stopping
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
    
    model.load_state_dict(best_model_state)
    torch.save(best_model_state, 'Models/TCN_model_weights.pth')

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
    plt.savefig('TCN_loss_curves.png')


def main():
    # Check if MPS is available, otherwise fallback to CPU or CUDA
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Load and prepare data
    df = load_and_prepare_data()
    
    # Select features for the model
    feature_columns = [
        'Close', 'EMA_8', 'SMA_200',
        'RSI', 'MACD', 'Signal_Line', 'MACD_Histogram',
        'BB_upper', 'BB_middle', 'BB_lower',
        'Volume',
        'sentiment_positive', 'sentiment_neutral', 'sentiment_negative'
    ]
    
    # Verify which columns are actually available
    available_columns = [col for col in feature_columns if col in df.columns]
    missing_columns = set(feature_columns) - set(available_columns)
    if missing_columns:
        print(f"Warning: The following columns are missing and will be excluded: {missing_columns}")
        feature_columns = available_columns

    # Clean and convert
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)


    # Normalize the data
    data = df[feature_columns].values
    data_mean = data.mean(axis=0)
    data_std = data.std(axis=0)
    data_normalized = (data - data_mean) / data_std

    X, y = create_sequences(data_normalized, CONFIG["SEQUENCE_LENGTH"], predict_ahead=1)

    # Split into training and validation sets 
    split_idx = int(len(X) * CONFIG["TRAIN_SIZE_RATIO"])
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

    # train model
    # Create an instance of the improved model
    num_inputs = X_train_tensor.shape[1]  # Number of features
    num_channels = [64, 128, 256, 128]  # Deeper network with more channels
    model = ImprovedTemporalConvNet(num_inputs, num_channels, kernel_size=3, dropout=CONFIG["DROPOUT_RATE"])
    model = model.to(device)

    # Define loss function and optimizer with weight decay
    criterion = nn.HuberLoss(delta=1.0)  # Huber loss is more robust to outliers than MSE
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["LEARNING_RATE"], weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=70)

    # Training loop with early stopping
    num_epochs = CONFIG["EPOCHS"]  # More epochs with early stopping
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
    # torch.save(best_model_state, 'Models/model_weights.pth')

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
    plt.savefig('TCN_loss_curves.png')
    # trainModel(device=device, X_train_tensor=X_train_tensor, X_val_tensor=X_val_tensor, y_train_tensor=y_train_tensor, y_val_tensor=y_val_tensor)
    
    # # load model 
    # num_inputs = X_train_tensor.shape[1]  # Number of features
    # num_channels = [64, 128, 256, 128] 
    # model = ImprovedTemporalConvNet(num_inputs, num_channels, kernel_size=3, dropout=CONFIG["DROPOUT_RATE"])
    # model = model.to(device)
    # model.load_state_dict(torch.load('Models/model_weights.pth', weights_only=False))
    # # model = torch.load('Models/model_weights.pth', weights_only)

    # Make predictions with the best model
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_val_tensor)
        test_predictions = test_predictions.cpu().numpy()

    # Denormalize predictions
    test_predictions_denormalized = test_predictions * data_std[0] + data_mean[0]

    # Denormalize truth values
    y_val_denormalized = y_val_original * data_std[0] + data_mean[0]

    # Confidence and Signals
    test_signals, confidence = generate_trading_signals(test_predictions_denormalized, y_val_denormalized, 
                                                      CONFIG['CONFIDENCE_THRESHOLD'])
    
    buy_signals = test_signals == 1
    sell_signals = test_signals == -1

    # Get the corresponding dates for validation set
    val_dates = df.index[split_idx + CONFIG["SEQUENCE_LENGTH"]:split_idx + CONFIG["SEQUENCE_LENGTH"] + len(y_val)]

    # Plot the predictions against the true values
    plt.figure(figsize=(14, 7))
    plt.plot(val_dates, y_val_denormalized, label="True Close Prices", color="blue", marker='o', markersize=3, linestyle="-")
    plt.plot(val_dates, test_predictions_denormalized, label="Predicted Close Prices", color="red", marker='x', markersize=3, linestyle="--")
    
    # Add Signals
    plt.scatter(df.index[-len(y_val_denormalized):][buy_signals], y_val_denormalized[buy_signals], 
               color='green', marker='^', s=100, label='Buy Signal')
    plt.scatter(df.index[-len(y_val_denormalized):][sell_signals], y_val_denormalized[sell_signals], 
               color='red', marker='v', s=100, label='Sell Signal')

    # Add EMA and SMA lines
    plt.plot(df.index[-len(y_val_denormalized):], df['EMA_8'].iloc[-len(y_val_denormalized):], 
            label='8-day EMA', color='purple', linestyle='--', alpha=0.7)
    plt.plot(df.index[-len(y_val_denormalized):], df['SMA_200'].iloc[-len(y_val_denormalized):], 
            label='200-day SMA', color='gray', linestyle='--', alpha=0.7)
    

    plt.title("Apple Stock: True vs. Predicted Close Prices (Improved Model)")
    plt.xlabel("Date")
    plt.ylabel("Close Price ($)")
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
    plt.savefig('TCN_stock_prediction.png')




    # Next day prediction
    model.eval()
    last_sequence, __ = create_sequences(-data_normalized[CONFIG["SEQUENCE_LENGTH"]:], CONFIG["SEQUENCE_LENGTH"], predict_ahead=1)
    last_sequence_tensor = torch.tensor(last_sequence, dtype=torch.float32).to(device)
    # Permute X tensors for convolution
    last_sequence_tensor = last_sequence_tensor.permute(0, 2, 1)
    with torch.no_grad():
        next_day_pred = model(last_sequence_tensor)
        next_day_pred = next_day_pred.cpu().numpy()


    # Denormalize Next day predictions
    next_day_pred = next_day_pred * data_std[0] + data_mean[0]

    last_actual = df['Close'].iloc[-1]
    confidence_next = calculate_confidence(next_day_pred.flatten(), np.array([last_actual]).flatten())
    
    print(f"\nPredicted price for next day: ${next_day_pred[0]}")
    print(f"Confidence level: {confidence_next[0]}")
    
    if confidence_next[0] >= CONFIG['CONFIDENCE_THRESHOLD']:
        if next_day_pred[0][0] > last_actual:
            print("High confidence BUY signal")
        else:
            print("High confidence SELL signal")
    else:
        print("No trading signal - confidence below threshold")

if __name__ == "__main__":
    main()

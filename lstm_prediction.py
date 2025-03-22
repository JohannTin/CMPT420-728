@ -4,9 +4,28 @@ from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import time
from tqdm import tqdm
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Configuration
CONFIG = {
    'SYMBOL': 'AAPL',
    'SEQUENCE_LENGTHS': [30, 60, 90],  # Different sequence lengths to try
    'TRAIN_SIZE_RATIO': 0.8,  # 80% of data for training
    'EPOCHS': 50,
    'BATCH_SIZES': [16, 32, 64],  # Different batch sizes to try
    'LSTM_UNITS': [25, 50, 100],  # Different LSTM units to try
    'DROPOUT_RATES': [0.1, 0.2, 0.3],  # Different dropout rates to try
    'MODEL_OPTIMIZER': 'adam',
    'MODEL_LOSS': 'mse',
    'CONFIDENCE_THRESHOLD': 0.95,  # Confidence threshold for trading signals
    'START_DATE': '2020-01-01',  # Start date for analysis (YYYY-MM-DD)
    'END_DATE': '2025-01-01'     # End date for analysis (YYYY-MM-DD)
}

def create_sequences(data, seq_length):
    """Create sequences for LSTM model"""
def create_sequences(data, seq_length):
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

def build_lstm_model(seq_length, n_features):
    """Build and return LSTM model"""
def build_lstm_model(seq_length, n_features, lstm_units, dropout_rate):
    """Build and return LSTM model with specified parameters"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        LSTM(lstm_units, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.compile(optimizer=CONFIG['MODEL_OPTIMIZER'], loss=CONFIG['MODEL_LOSS'])
    return model

def train_model_with_params(model, X_train, y_train, X_test, y_test, epochs, batch_size):
    """Train a model and return its history"""
    n_batches = len(X_train) // batch_size
    history = {'loss': [], 'val_loss': []}
    
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            batch_x = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            loss = model.train_on_batch(batch_x, batch_y)
            epoch_loss += loss
        
        val_loss = model.evaluate(X_test, y_test, verbose=0)
        history['loss'].append(epoch_loss / n_batches)
        history['val_loss'].append(val_loss)
    
    return history

def find_best_hyperparameters(scaled_data, train_size):
    """Find the best hyperparameters by training models with different combinations"""
    best_val_loss = float('inf')
    best_params = None
    best_model = None
    best_history = None
    
    print("\nPerforming hyperparameter tuning...")
    total_combinations = (len(CONFIG['SEQUENCE_LENGTHS']) * 
                         len(CONFIG['BATCH_SIZES']) * 
                         len(CONFIG['LSTM_UNITS']) * 
                         len(CONFIG['DROPOUT_RATES']))
    
    with tqdm(total=total_combinations, desc="Testing hyperparameters") as pbar:
        for seq_length in CONFIG['SEQUENCE_LENGTHS']:
            # Create sequences for current sequence length
            X, y = create_sequences(scaled_data, seq_length)
            X_train, X_test = X[:train_size], X[train_size:]
            y_train, y_test = y[:train_size], y[train_size:]
            
            for batch_size in CONFIG['BATCH_SIZES']:
                for lstm_units in CONFIG['LSTM_UNITS']:
                    for dropout_rate in CONFIG['DROPOUT_RATES']:
                        # Build and train model
                        model = build_lstm_model(seq_length, 1, lstm_units, dropout_rate)
                        history = train_model_with_params(
                            model, X_train, y_train, X_test, y_test,
                            CONFIG['EPOCHS'], batch_size
                        )
                        
                        # Get the best validation loss
                        min_val_loss = min(history['val_loss'])
                        
                        if min_val_loss < best_val_loss:
                            best_val_loss = min_val_loss
                            best_params = {
                                'sequence_length': seq_length,
                                'batch_size': batch_size,
                                'lstm_units': lstm_units,
                                'dropout_rate': dropout_rate
                            }
                            best_model = model
                            best_history = history
                        
                        pbar.update(1)
                        pbar.set_postfix({
                            'best_val_loss': f'{best_val_loss:.4f}',
                            'seq_len': seq_length,
                            'batch': batch_size,
                            'units': lstm_units,
                            'dropout': dropout_rate
                        })
    
    print("\nBest hyperparameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return best_model, best_history, best_params

def calculate_confidence(predictions, actual_values):
    """Calculate confidence level based on prediction error"""
    # Ensure inputs are 1D arrays
    predictions = predictions.flatten()
    actual_values = actual_values.flatten()
    
    errors = np.abs(predictions - actual_values)
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    
    # Calculate confidence as inverse of normalized error
    confidence = 1 - (errors / (mean_error + 2 * std_error))
    confidence = np.clip(confidence, 0, 1)
    return confidence

def generate_trading_signals(predictions, actual_values, confidence_threshold):
    """Generate trading signals based on predictions and confidence"""
    # Ensure inputs are 1D arrays
    predictions = predictions.flatten()
    actual_values = actual_values.flatten()
    
    confidence = calculate_confidence(predictions, actual_values)
    signals = np.zeros_like(predictions)
    
    # Generate signals only when confidence is high
    high_confidence_mask = confidence >= confidence_threshold
    price_change = predictions - actual_values
    
    # Buy signal when price is predicted to increase with high confidence
    signals[high_confidence_mask & (price_change > 0)] = 1
    # Sell signal when price is predicted to decrease with high confidence
    signals[high_confidence_mask & (price_change < 0)] = -1
    
    return signals, confidence

def main():
    try:
        # Initialize the TimeSeries class with your API key
        ts = TimeSeries(key='Z3WPH5FVFZ3EW8Y1')
        print("Loading stock data from local file...")
        # Read data from local CSV file
        df = pd.read_csv('aapl_data.csv')
        
        print("Fetching stock data...")
        # Get daily data for AAPL
        data, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')
        # Convert date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        
        # Convert to pandas DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        # Filter data by date range
        start_date = pd.to_datetime(CONFIG['START_DATE'])
        end_date = pd.to_datetime(CONFIG['END_DATE'])
        df = df[(df.index >= start_date) & (df.index <= end_date)]
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Convert string values to float
        for col in tqdm(['Open', 'High', 'Low', 'Close', 'Volume'], desc="Processing columns"):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by date
        df = df.sort_index()
        if len(df) == 0:
            raise ValueError("No data available for the specified date range")
            
        print(f"Analyzing data from {start_date.date()} to {end_date.date()}")
        
        # Use only the closing prices for prediction
        data = df['Close'].values.reshape(-1, 1)
@ -60,47 +193,20 @@ def main():
        scaled_data = scaler.fit_transform(data)
        
        # Parameters
        sequence_length = 60  # Number of days to look back
        train_size = int(len(scaled_data) * 0.8)
        train_size = int(len(scaled_data) * CONFIG['TRAIN_SIZE_RATIO'])
        
        # Create sequences
        X, y = create_sequences(scaled_data, sequence_length)
        # Find best hyperparameters and get the best model
        best_model, history, best_params = find_best_hyperparameters(scaled_data, train_size)
        
        # Split into train and test sets
        # Create sequences with best sequence length
        X, y = create_sequences(scaled_data, best_params['sequence_length'])
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Build and train the model
        model = build_lstm_model(sequence_length, 1)
        print("\nTraining model...")
        
        # Custom training loop with tqdm
        epochs = 5
        batch_size = 32
        n_batches = len(X_train) // batch_size
        history = {'loss': [], 'val_loss': []}
        
        for epoch in tqdm(range(epochs), desc="Training epochs"):
            epoch_loss = 0
            for i in range(n_batches):
                start_idx = i * batch_size
                end_idx = start_idx + batch_size
                batch_x = X_train[start_idx:end_idx]
                batch_y = y_train[start_idx:end_idx]
                
                loss = model.train_on_batch(batch_x, batch_y)
                epoch_loss += loss
            
            # Calculate validation loss
            val_loss = model.evaluate(X_test, y_test, verbose=0)
            
            history['loss'].append(epoch_loss / n_batches)
            history['val_loss'].append(val_loss)
        
        print("\nMaking predictions...")
        # Make predictions
        train_predictions = model.predict(X_train, verbose=0)
        test_predictions = model.predict(X_test, verbose=0)
        print("\nMaking predictions with best model...")
        # Make predictions using the best model
        train_predictions = best_model.predict(X_train, verbose=0)
        test_predictions = best_model.predict(X_test, verbose=0)
        
        # Inverse transform predictions
        train_predictions = scaler.inverse_transform(train_predictions)
@ -108,6 +214,13 @@ def main():
        test_predictions = scaler.inverse_transform(test_predictions)
        y_test_inv = scaler.inverse_transform(y_test)
        
        # Generate trading signals
        test_signals, confidence = generate_trading_signals(
            test_predictions, 
            y_test_inv, 
            CONFIG['CONFIDENCE_THRESHOLD']
        )
        
        # Calculate RMSE
        train_rmse = np.sqrt(np.mean((train_predictions - y_train_inv) ** 2))
        test_rmse = np.sqrt(np.mean((test_predictions - y_test_inv) ** 2))
@ -115,11 +228,34 @@ def main():
        print(f"Test RMSE: {test_rmse:.2f}")
        
        print("\nGenerating plots...")
        # Plot the results
        # Plot the results with trading signals
        plt.figure(figsize=(15, 7))
        plt.plot(df.index[-len(y_test_inv):], y_test_inv, label='Actual')
        plt.plot(df.index[-len(test_predictions):], test_predictions, label='Predicted')
        plt.title('AAPL Stock Price Prediction')
        plt.plot(df.index[-len(y_test_inv):], y_test_inv, label='Actual', color='blue')
        plt.plot(df.index[-len(test_predictions):], test_predictions, label='Predicted', color='orange')
        
        # Plot buy signals
        buy_signals = test_signals == 1
        sell_signals = test_signals == -1
        
        plt.scatter(
            df.index[-len(y_test_inv):][buy_signals],
            y_test_inv[buy_signals],
            color='green',
            marker='^',
            s=100,
            label='Buy Signal'
        )
        
        plt.scatter(
            df.index[-len(y_test_inv):][sell_signals],
            y_test_inv[sell_signals],
            color='red',
            marker='v',
            s=100,
            label='Sell Signal'
        )
        
        plt.title('AAPL Stock Price Prediction with Trading Signals')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
@ -141,19 +277,37 @@ def main():
        plt.savefig('training_history.png')
        print("\nTraining history plot has been saved as 'training_history.png'")
        
        # Predict next day
        last_sequence = scaled_data[-sequence_length:]
        last_sequence = last_sequence.reshape((1, sequence_length, 1))
        next_day_pred = model.predict(last_sequence, verbose=0)
        # Predict next day with confidence
        last_sequence = scaled_data[-best_params['sequence_length']:]
        last_sequence = last_sequence.reshape((1, best_params['sequence_length'], 1))
        next_day_pred = best_model.predict(last_sequence, verbose=0)
        next_day_pred = scaler.inverse_transform(next_day_pred)
        
        # Calculate confidence for next day prediction
        last_actual = data[-1]
        next_day_pred_flat = next_day_pred.flatten()
        last_actual_flat = np.array([last_actual])
        next_day_confidence = calculate_confidence(next_day_pred_flat, last_actual_flat)
        
        print(f"\nPredicted price for next day: ${next_day_pred[0][0]:.2f}")
        print(f"Confidence level: {next_day_confidence[0]:.2%}")
        
        if next_day_confidence[0] >= CONFIG['CONFIDENCE_THRESHOLD']:
            if next_day_pred[0][0] > last_actual:
                print("High confidence BUY signal")
            else:
                print("High confidence SELL signal")
        else:
            print("No trading signal - confidence below threshold")
        
        print("\nSaving predictions...")
        # Save predictions to CSV
        # Save predictions and signals to CSV
        predictions_df = pd.DataFrame({
            'Date': df.index[-len(test_predictions):],
            'Actual': y_test_inv.flatten(),
            'Predicted': test_predictions.flatten()
            'Predicted': test_predictions.flatten(),
            'Signal': test_signals.flatten(),
            'Confidence': confidence.flatten()
        })
        predictions_df.to_csv('lstm_predictions.csv')
        print("\nPredictions have been saved to 'lstm_predictions.csv'")

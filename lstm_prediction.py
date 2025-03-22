import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from alpha_vantage.timeseries import TimeSeries
import time
from tqdm import tqdm

def create_sequences(data, seq_length):
    """Create sequences for LSTM model"""
    sequences = []
    targets = []
    for i in tqdm(range(len(data) - seq_length), desc="Creating sequences"):
        sequences.append(data[i:(i + seq_length)])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

def build_lstm_model(seq_length, n_features):
    """Build and return LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def main():
    try:
        # Initialize the TimeSeries class with your API key
        ts = TimeSeries(key='Z3WPH5FVFZ3EW8Y1')
        
        print("Fetching stock data...")
        # Get daily data for AAPL
        data, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')
        
        # Convert to pandas DataFrame
        df = pd.DataFrame.from_dict(data, orient='index')
        df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        # Convert index to datetime
        df.index = pd.to_datetime(df.index)
        
        # Convert string values to float
        for col in tqdm(['Open', 'High', 'Low', 'Close', 'Volume'], desc="Processing columns"):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort by date
        df = df.sort_index()
        
        # Use only the closing prices for prediction
        data = df['Close'].values.reshape(-1, 1)
        
        # Normalize the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Parameters
        sequence_length = 60  # Number of days to look back
        train_size = int(len(scaled_data) * 0.8)
        
        # Create sequences
        X, y = create_sequences(scaled_data, sequence_length)
        
        # Split into train and test sets
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
        
        # Inverse transform predictions
        train_predictions = scaler.inverse_transform(train_predictions)
        y_train_inv = scaler.inverse_transform(y_train)
        test_predictions = scaler.inverse_transform(test_predictions)
        y_test_inv = scaler.inverse_transform(y_test)
        
        # Calculate RMSE
        train_rmse = np.sqrt(np.mean((train_predictions - y_train_inv) ** 2))
        test_rmse = np.sqrt(np.mean((test_predictions - y_test_inv) ** 2))
        print(f"\nTrain RMSE: {train_rmse:.2f}")
        print(f"Test RMSE: {test_rmse:.2f}")
        
        print("\nGenerating plots...")
        # Plot the results
        plt.figure(figsize=(15, 7))
        plt.plot(df.index[-len(y_test_inv):], y_test_inv, label='Actual')
        plt.plot(df.index[-len(test_predictions):], test_predictions, label='Predicted')
        plt.title('AAPL Stock Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('lstm_prediction_plot.png')
        print("\nPrediction plot has been saved as 'lstm_prediction_plot.png'")
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history['loss'], label='Training Loss')
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
        print("\nTraining history plot has been saved as 'training_history.png'")
        
        # Predict next day
        last_sequence = scaled_data[-sequence_length:]
        last_sequence = last_sequence.reshape((1, sequence_length, 1))
        next_day_pred = model.predict(last_sequence, verbose=0)
        next_day_pred = scaler.inverse_transform(next_day_pred)
        print(f"\nPredicted price for next day: ${next_day_pred[0][0]:.2f}")
        
        print("\nSaving predictions...")
        # Save predictions to CSV
        predictions_df = pd.DataFrame({
            'Date': df.index[-len(test_predictions):],
            'Actual': y_test_inv.flatten(),
            'Predicted': test_predictions.flatten()
        })
        predictions_df.to_csv('lstm_predictions.csv')
        print("\nPredictions have been saved to 'lstm_predictions.csv'")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print("Please check your internet connection and try again.")

if __name__ == "__main__":
    main() 
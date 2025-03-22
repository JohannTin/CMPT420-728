import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# Configuration
CONFIG = {
    'SYMBOL': 'AAPL',
    'SEQUENCE_LENGTHS': [30, 60],
    'TRAIN_SIZE_RATIO': 0.8,
    'EPOCHS': 5,
    'BATCH_SIZES': [16, 32],
    'LSTM_UNITS': [25, 50],
    'DROPOUT_RATES': [0.1, 0.2],
    'MODEL_OPTIMIZER': 'adam',
    'MODEL_LOSS': 'mse',
    'CONFIDENCE_THRESHOLD': 0.95,
    'START_DATE': '2020-01-01',
    'END_DATE': '2025-01-01'
}

def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        targets.append(data[i + seq_length])
    return np.array(sequences), np.array(targets)

def build_lstm_model(seq_length, n_features, lstm_units, dropout_rate):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=(seq_length, n_features)),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(1)
    ])
    model.compile(optimizer=CONFIG['MODEL_OPTIMIZER'], loss=CONFIG['MODEL_LOSS'])
    return model

def calculate_confidence(predictions, actuals):
    """Calculate confidence based on prediction error."""
    errors = np.abs(predictions - actuals.flatten())
    max_error = np.max(actuals) - np.min(actuals)
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

def find_best_hyperparameters(data, train_size):
    """Find the best hyperparameters for the LSTM model."""
    best_val_loss = float('inf')
    best_model = None
    best_history = None
    best_params = None
    
    for seq_length in CONFIG['SEQUENCE_LENGTHS']:
        X, y = create_sequences(data, seq_length)
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        for batch_size in CONFIG['BATCH_SIZES']:
            for lstm_units in CONFIG['LSTM_UNITS']:
                for dropout_rate in CONFIG['DROPOUT_RATES']:
                    print(f"Testing: seq_length={seq_length}, batch_size={batch_size}, "
                          f"lstm_units={lstm_units}, dropout_rate={dropout_rate}")
                    
                    model = build_lstm_model(seq_length, X.shape[2], lstm_units, dropout_rate)
                    history = model.fit(
                        X_train, y_train,
                        epochs=CONFIG['EPOCHS'],
                        batch_size=batch_size,
                        validation_data=(X_val, y_val),
                        verbose=0
                    )
                    
                    val_loss = min(history.history['val_loss'])
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        best_model = model
                        best_history = history
                        best_params = {
                            'sequence_length': seq_length,
                            'batch_size': batch_size,
                            'lstm_units': lstm_units,
                            'dropout_rate': dropout_rate
                        }
                        
                    print(f"Validation loss: {val_loss:.6f}")
    
    print("\nBest parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    return best_model, best_history, best_params

def main():
    # Load data
    print("Loading data from aapl_data.csv...")
    scaler = MinMaxScaler()
    df = pd.read_csv('aapl_data.csv', index_col=0)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[(df.index >= CONFIG['START_DATE']) & (df.index <= CONFIG['END_DATE'])]

    # Clean and convert
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df.dropna(inplace=True)

    data = df['Close'].values.reshape(-1, 1)
    scaled_data = scaler.fit_transform(data)
    train_size = int(len(scaled_data) * CONFIG['TRAIN_SIZE_RATIO'])

    # Tune and get best model
    best_model, history, best_params = find_best_hyperparameters(scaled_data, train_size)

    # Final training and prediction
    X, y = create_sequences(scaled_data, best_params['sequence_length'])
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Predictions
    best_model.fit(X_train, y_train, epochs=CONFIG['EPOCHS'], batch_size=best_params['batch_size'], verbose=0)
    train_predictions = best_model.predict(X_train, verbose=0)
    test_predictions = best_model.predict(X_test, verbose=0)

    train_predictions = scaler.inverse_transform(train_predictions)
    test_predictions = scaler.inverse_transform(test_predictions)
    y_test_inv = scaler.inverse_transform(y_test)

    # Confidence and Signals
    test_signals, confidence = generate_trading_signals(test_predictions, y_test_inv, CONFIG['CONFIDENCE_THRESHOLD'])

    # RMSE
    train_rmse = np.sqrt(np.mean((train_predictions - scaler.inverse_transform(y_train)) ** 2))
    test_rmse = np.sqrt(np.mean((test_predictions - y_test_inv) ** 2))
    print(f"Train RMSE: {train_rmse:.2f}, Test RMSE: {test_rmse:.2f}")

    # Plotting
    plt.figure(figsize=(15, 7))
    plt.plot(df.index[-len(y_test_inv):], y_test_inv, label='Actual', color='blue')
    plt.plot(df.index[-len(test_predictions):], test_predictions, label='Predicted', color='orange')

    buy_signals = test_signals == 1
    sell_signals = test_signals == -1
    plt.scatter(df.index[-len(y_test_inv):][buy_signals], y_test_inv[buy_signals], color='green', marker='^', s=100, label='Buy Signal')
    plt.scatter(df.index[-len(y_test_inv):][sell_signals], y_test_inv[sell_signals], color='red', marker='v', s=100, label='Sell Signal')

    plt.title('AAPL Stock Price Prediction with Trading Signals')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.savefig('training_history.png')

    # Next day prediction
    last_sequence = scaled_data[-best_params['sequence_length']:]
    last_sequence = last_sequence.reshape((1, best_params['sequence_length'], 1))
    next_day_pred = best_model.predict(last_sequence, verbose=0)
    next_day_pred = scaler.inverse_transform(next_day_pred)

    last_actual = data[-1]
    confidence_next = calculate_confidence(next_day_pred.flatten(), np.array([last_actual]))

    print(f"\nPredicted price for next day: ${next_day_pred[0][0]:.2f}")
    print(f"Confidence level: {confidence_next[0]:.2%}")

    if confidence_next[0] >= CONFIG['CONFIDENCE_THRESHOLD']:
        if next_day_pred[0][0] > last_actual:
            print("High confidence BUY signal")
        else:
            print("High confidence SELL signal")
    else:
        print("No trading signal - confidence below threshold")

    # Save predictions
    predictions_df = pd.DataFrame({
        'Date': df.index[-len(test_predictions):],
        'Actual': y_test_inv.flatten(),
        'Predicted': test_predictions.flatten(),
        'Signal': test_signals.flatten(),
        'Confidence': confidence.flatten()
    })
    predictions_df.to_csv('lstm_predictions.csv', index=False)
    print("Predictions saved to lstm_predictions.csv")

if __name__ == "__main__":
    main()

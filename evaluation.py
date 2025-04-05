import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

# Read the prediction files
lstm_df = pd.read_csv('lstm_predictions.csv')
tcn_df = pd.read_csv('Price_History_Model/TCN_predictions.csv')
tft_df = pd.read_csv('tft_predictions_improved.csv')

# Calculate metrics for LSTM
lstm_mae = mean_absolute_error(lstm_df['Actual'], lstm_df['Predicted'])
lstm_r2 = r2_score(lstm_df['Actual'], lstm_df['Predicted'])

# Calculate metrics for TCN
tcn_mae = mean_absolute_error(tcn_df['Actual_Close'], tcn_df['Predicted_Close'])
tcn_r2 = r2_score(tcn_df['Actual_Close'], tcn_df['Predicted_Close'])

# Calculate metrics for TFT
tft_mae = mean_absolute_error(tft_df['Actual'], tft_df['Predicted_Prob'])
tft_r2 = r2_score(tft_df['Actual'], tft_df['Predicted_Prob'])

# Print results
print("Model Evaluation Results:")
print("\nLSTM Model:")
print(f"MAE: {lstm_mae:.4f}")
print(f"R-squared: {lstm_r2:.4f}")

print("\nTCN Model:")
print(f"MAE: {tcn_mae:.4f}")
print(f"R-squared: {tcn_r2:.4f}")

print("\nTFT Model:")
print(f"MAE: {tft_mae:.4f}")
print(f"R-squared: {tft_r2:.4f}")

# Create a comparison table
results = pd.DataFrame({
    'Model': ['LSTM', 'TCN', 'TFT'],
    'MAE': [lstm_mae, tcn_mae, tft_mae],
    'R-squared': [lstm_r2, tcn_r2, tft_r2]
})

print("\nComparison Table:")
print(results.to_string(index=False))

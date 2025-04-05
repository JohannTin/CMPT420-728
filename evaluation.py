import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_theme()

# Read the prediction files
lstm_df = pd.read_csv('lstm_predictions.csv')
tcn_df = pd.read_csv('Price_History_Model/TCN_predictions.csv')
tft_df = pd.read_csv('tft_predictions.csv')

# Convert date columns to datetime
lstm_df['Date'] = pd.to_datetime(lstm_df['Date'])
tcn_df['Date'] = pd.to_datetime(tcn_df['Date'])
tft_df['Date'] = pd.to_datetime(tft_df['Date'])

# Calculate metrics for LSTM
lstm_mae = mean_absolute_error(lstm_df['Actual'], lstm_df['Predicted'])
lstm_r2 = r2_score(lstm_df['Actual'], lstm_df['Predicted'])

# Calculate metrics for TCN
tcn_mae = mean_absolute_error(tcn_df['Actual_Close'], tcn_df['Predicted_Close'])
tcn_r2 = r2_score(tcn_df['Actual_Close'], tcn_df['Predicted_Close'])

# Calculate metrics for TFT
tft_mae = mean_absolute_error(tft_df['Actual'], tft_df['Predicted'])
tft_r2 = r2_score(tft_df['Actual'], tft_df['Predicted'])

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

# Create visualizations
plt.figure(figsize=(20, 12))

# 1. Time Series Plot
plt.subplot(2, 2, 1)
plt.plot(lstm_df['Date'], lstm_df['Actual'], label='Actual', linewidth=2, alpha=0.7)
plt.plot(lstm_df['Date'], lstm_df['Predicted'], '-', label='LSTM', linewidth=2, alpha=0.7)
plt.plot(tcn_df['Date'], tcn_df['Predicted_Close'], '-', label='TCN', linewidth=2, alpha=0.7)
plt.plot(tft_df['Date'], tft_df['Predicted'], '-', label='TFT', linewidth=2, alpha=0.7)
plt.title('Stock Price Predictions Comparison')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

# 2. MAE Comparison
plt.subplot(2, 2, 2)
sns.barplot(x='Model', y='MAE', data=results)
plt.title('Mean Absolute Error (MAE) Comparison')
plt.ylabel('MAE (Lower is better)')

# 3. Actual vs Predicted Scatter Plots
plt.subplot(2, 2, 3)
plt.scatter(lstm_df['Actual'], lstm_df['Predicted'], alpha=0.5, label='LSTM')
plt.scatter(tcn_df['Actual_Close'], tcn_df['Predicted_Close'], alpha=0.5, label='TCN')
plt.scatter(tft_df['Actual'], tft_df['Predicted'], alpha=0.5, label='TFT')
plt.plot([min(lstm_df['Actual']), max(lstm_df['Actual'])], 
         [min(lstm_df['Actual']), max(lstm_df['Actual'])], 
         'r--', label='Perfect Prediction')
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()

# 4. Residual Plot
plt.subplot(2, 2, 4)
lstm_residuals = lstm_df['Actual'] - lstm_df['Predicted']
tcn_residuals = tcn_df['Actual_Close'] - tcn_df['Predicted_Close']
tft_residuals = tft_df['Actual'] - tft_df['Predicted']

plt.scatter(lstm_df['Actual'], lstm_residuals, alpha=0.5, label='LSTM')
plt.scatter(tcn_df['Actual_Close'], tcn_residuals, alpha=0.5, label='TCN')
plt.scatter(tft_df['Actual'], tft_residuals, alpha=0.5, label='TFT')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.legend()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Create individual time series plots for better visibility
plt.figure(figsize=(15, 10))

# LSTM Predictions
plt.subplot(3, 1, 1)
plt.plot(lstm_df['Date'], lstm_df['Actual'], label='Actual', linewidth=2)
plt.plot(lstm_df['Date'], lstm_df['Predicted'], '-', label='LSTM Predicted', linewidth=2)
plt.title('LSTM Model Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

# TCN Predictions
plt.subplot(3, 1, 2)
plt.plot(tcn_df['Date'], tcn_df['Actual_Close'], label='Actual', linewidth=2)
plt.plot(tcn_df['Date'], tcn_df['Predicted_Close'], '-', label='TCN Predicted', linewidth=2)
plt.title('TCN Model Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

# TFT Predictions
plt.subplot(3, 1, 3)
plt.plot(tft_df['Date'], tft_df['Actual'], label='Actual', linewidth=2)
plt.plot(tft_df['Date'], tft_df['Predicted'], '-', label='TFT Predicted', linewidth=2)
plt.title('TFT Model Predictions')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.savefig('individual_predictions.png', dpi=300, bbox_inches='tight')
plt.close()
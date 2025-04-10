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

# Filter data for 2024 only
start_date = '2024-01-01'
end_date = '2024-12-31'

lstm_df = lstm_df[(lstm_df['Date'] >= start_date) & (lstm_df['Date'] <= end_date)]
tcn_df = tcn_df[(tcn_df['Date'] >= start_date) & (tcn_df['Date'] <= end_date)]
tft_df = tft_df[(tft_df['Date'] >= start_date) & (tft_df['Date'] <= end_date)]

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
print("Model Evaluation Results (2024):")
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

# 1. Time Series Plot
plt.figure(figsize=(15, 8))
plt.plot(lstm_df['Date'], lstm_df['Actual'], label='Actual', linewidth=2, alpha=0.7)
plt.plot(lstm_df['Date'], lstm_df['Predicted'], '-', label='LSTM', linewidth=2, alpha=0.7)
plt.plot(tcn_df['Date'], tcn_df['Predicted_Close'], '-', label='TCN', linewidth=2, alpha=0.7)
plt.plot(tft_df['Date'], tft_df['Predicted'], '-', label='TFT', linewidth=2, alpha=0.7)
plt.title('Stock Price Predictions Comparison (2024)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('time_series_comparison_2024.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. MAE Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MAE', data=results)
plt.title('Mean Absolute Error (MAE) Comparison (2024)')
plt.ylabel('MAE (Lower is better)')
plt.tight_layout()
plt.savefig('mae_comparison_2024.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Actual vs Predicted Scatter Plots
plt.figure(figsize=(10, 8))
plt.scatter(lstm_df['Actual'], lstm_df['Predicted'], alpha=0.5, label='LSTM')
plt.scatter(tcn_df['Actual_Close'], tcn_df['Predicted_Close'], alpha=0.5, label='TCN')
plt.scatter(tft_df['Actual'], tft_df['Predicted'], alpha=0.5, label='TFT')
plt.plot([min(lstm_df['Actual']), max(lstm_df['Actual'])], 
         [min(lstm_df['Actual']), max(lstm_df['Actual'])], 
         'r--', label='Perfect Prediction')
plt.title('Actual vs Predicted Values (2024)')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.legend()
plt.tight_layout()
plt.savefig('scatter_comparison_2024.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Residual Plot
plt.figure(figsize=(10, 8))
lstm_residuals = lstm_df['Actual'] - lstm_df['Predicted']
tcn_residuals = tcn_df['Actual_Close'] - tcn_df['Predicted_Close']
tft_residuals = tft_df['Actual'] - tft_df['Predicted']

plt.scatter(lstm_df['Actual'], lstm_residuals, alpha=0.5, label='LSTM')
plt.scatter(tcn_df['Actual_Close'], tcn_residuals, alpha=0.5, label='TCN')
plt.scatter(tft_df['Actual'], tft_residuals, alpha=0.5, label='TFT')
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residual Plot (2024)')
plt.xlabel('Actual Values')
plt.ylabel('Residuals')
plt.legend()
plt.tight_layout()
plt.savefig('residual_comparison_2024.png', dpi=300, bbox_inches='tight')
plt.close()


# LSTM Predictions
plt.figure(figsize=(15, 6))
plt.plot(lstm_df['Date'], lstm_df['Actual'], label='Actual', linewidth=2)
plt.plot(lstm_df['Date'], lstm_df['Predicted'], '-', label='LSTM Predicted', linewidth=2)
plt.title('LSTM Model Predictions (2024)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('lstm_predictions_2024.png', dpi=300, bbox_inches='tight')
plt.close()

# TCN Predictions
plt.figure(figsize=(15, 6))
plt.plot(tcn_df['Date'], tcn_df['Actual_Close'], label='Actual', linewidth=2)
plt.plot(tcn_df['Date'], tcn_df['Predicted_Close'], '-', label='TCN Predicted', linewidth=2)
plt.title('TCN Model Predictions (2024)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('tcn_predictions_2024.png', dpi=300, bbox_inches='tight')
plt.close()

# TFT Predictions
plt.figure(figsize=(15, 6))
plt.plot(tft_df['Date'], tft_df['Actual'], label='Actual', linewidth=2)
plt.plot(tft_df['Date'], tft_df['Predicted'], '-', label='TFT Predicted', linewidth=2)
plt.title('TFT Model Predictions (2024)')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.savefig('tft_predictions_2024.png', dpi=300, bbox_inches='tight')
plt.close()
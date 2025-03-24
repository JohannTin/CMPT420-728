import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('aapl_data.csv')

# Calculate Moving Averages
df['SMA_50'] = df['Close'].rolling(window=50).mean()
df['SMA_200'] = df['Close'].rolling(window=200).mean()
df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()

# Calculate MACD
df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
df['MACD'] = df['EMA_12'] - df['EMA_26']
df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']

# Calculate RSI
delta = df['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['RSI'] = 100 - (100 / (1 + rs))

# Calculate Stochastic Oscillator
df['14-high'] = df['High'].rolling(14).max()
df['14-low'] = df['Low'].rolling(14).min()
df['%K'] = (df['Close'] - df['14-low']) * 100 / (df['14-high'] - df['14-low'])
df['%D'] = df['%K'].rolling(3).mean()

# Calculate Bollinger Bands
df['BB_middle'] = df['Close'].rolling(window=20).mean()
df['BB_upper'] = df['BB_middle'] + 2 * df['Close'].rolling(window=20).std()
df['BB_lower'] = df['BB_middle'] - 2 * df['Close'].rolling(window=20).std()

# Calculate ATR
df['TR'] = np.maximum(
    np.maximum(
        df['High'] - df['Low'],
        abs(df['High'] - df['Close'].shift(1))
    ),
    abs(df['Low'] - df['Close'].shift(1))
)
df['ATR'] = df['TR'].rolling(window=14).mean()

# Calculate OBV
df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()

# Create subplots
fig, axes = plt.subplots(4, 1, figsize=(15, 20), gridspec_kw={'height_ratios': [3, 1, 1, 1]})
fig.suptitle('AAPL Technical Analysis', fontsize=16)

# Plot 1: Price and Moving Averages + Bollinger Bands
axes[0].plot(df.index, df['Close'], label='AAPL Close Price', color='blue', alpha=0.7)
axes[0].plot(df.index, df['SMA_50'], label='SMA 50', color='orange')
axes[0].plot(df.index, df['SMA_200'], label='SMA 200', color='red')
axes[0].plot(df.index, df['EMA_8'], label='EMA 8', color='purple')
axes[0].plot(df.index, df['BB_upper'], label='BB Upper', color='gray', linestyle='--')
axes[0].plot(df.index, df['BB_middle'], label='BB Middle', color='gray', linestyle='-')
axes[0].plot(df.index, df['BB_lower'], label='BB Lower', color='gray', linestyle='--')
axes[0].set_ylabel('Price (USD)')
axes[0].legend()
axes[0].grid(True)

# Plot 2: MACD
axes[1].plot(df.index, df['MACD'], label='MACD', color='blue')
axes[1].plot(df.index, df['Signal_Line'], label='Signal Line', color='red')
axes[1].bar(df.index, df['MACD_Histogram'], label='MACD Histogram', color='gray', alpha=0.3)
axes[1].set_ylabel('MACD')
axes[1].legend()
axes[1].grid(True)

# Plot 3: RSI and Stochastic
axes[2].plot(df.index, df['RSI'], label='RSI', color='purple')
axes[2].plot(df.index, df['%K'], label='Stochastic %K', color='blue')
axes[2].plot(df.index, df['%D'], label='Stochastic %D', color='red')
axes[2].axhline(y=70, color='r', linestyle='--', alpha=0.3)
axes[2].axhline(y=30, color='g', linestyle='--', alpha=0.3)
axes[2].set_ylabel('RSI/Stochastic')
axes[2].legend()
axes[2].grid(True)

# Plot 4: Volume and OBV
ax4_1 = axes[3]
ax4_2 = ax4_1.twinx()
ax4_1.bar(df.index, df['Volume'], label='Volume', color='blue', alpha=0.3)
ax4_2.plot(df.index, df['OBV'], label='OBV', color='orange')
ax4_1.set_ylabel('Volume')
ax4_2.set_ylabel('OBV')
lines1, labels1 = ax4_1.get_legend_handles_labels()
lines2, labels2 = ax4_2.get_legend_handles_labels()
ax4_1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
ax4_1.grid(True)

# Adjust layout and save
plt.xlabel('Days')
plt.tight_layout()
plt.savefig('aapl_indicators_plot.png')
plt.close()

# Save the updated dataframe to a new CSV file
df.to_csv('appl_data_with_indicators.csv', index=False)

print("All technical indicators have been calculated and saved to 'appl_data_with_indicators.csv'")
print("Plot has been saved as 'aapl_indicators_plot.png'")

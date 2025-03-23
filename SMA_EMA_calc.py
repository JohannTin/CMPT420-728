import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('aapl_data.csv')

# Calculate 8-period EMA
df['EMA_8'] = df['Close'].ewm(span=8, adjust=False).mean()

# Calculate 200-period SMA
df['SMA_200'] = df['Close'].rolling(window=200).mean()

# Create the plot
plt.figure(figsize=(12, 6))
plt.plot(df['Close'], label='AAPL Close Price', color='blue')
plt.plot(df['EMA_8'], label='8-period EMA', color='red')
plt.plot(df['SMA_200'], label='200-period SMA', color='green')

plt.title('AAPL Stock Price with EMA-8 and SMA-200')
plt.xlabel('Days')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True)

# Save the plot
plt.savefig('aapl_indicators_plot.png')
plt.close()

# Save the updated dataframe to a new CSV file
df.to_csv('appl_data_with_indicators.csv', index=False)

print("EMA-8 and SMA-200 have been calculated and saved to 'appl_data_with_indicators.csv'")
print("Plot has been saved as 'aapl_indicators_plot.png'")

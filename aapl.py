from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time
import matplotlib.pyplot as plt

# https://www.alphavantage.co/documentation/
try:
    # Initialize the TimeSeries class with your API key
    API_KEY = 'IB4I6GNFN8W571ZP'
    ts = TimeSeries(key=API_KEY)
    
    # Get daily data for AAPL
    # outputsize can be 'compact' (last 100 data points) or 'full' (complete history)
    data, meta_data = ts.get_daily(symbol='AAPL', outputsize='full')
    
    # Convert to pandas DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    
    # Display date range
    print("\nDate Range:")
    print(f"Start Date: {df.index.min()}")
    print(f"End Date: {df.index.max()}")
    
    # Convert string values to float
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by date
    df = df.sort_index()
    
    # Calculate 8 EMA and 200 SMA
    df['8_EMA'] = df['Close'].ewm(span=8, adjust=False).mean()
    df['200_SMA'] = df['Close'].rolling(window=200).mean()
    
    # Validate data
    if df.empty:
        raise ValueError("No data was retrieved from the API")
    
    if df.isnull().any().any():
        print("Warning: Some data points are missing")
    
    # Display the first few rows
    print("\nFirst few rows of the data:")
    print(df.head())
    
    # Display summary statistics
    print("\nSummary Statistics:")
    print(df.describe())
    
    # Display basic information about the dataset
    print("\nDataset Info:")
    print(df.info())
    
    # Create a plot of closing prices with EMA and SMA
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Closing Price', alpha=0.8)
    plt.plot(df.index, df['8_EMA'], label='8-day EMA', alpha=0.8)
    plt.plot(df.index, df['200_SMA'], label='200-day SMA', alpha=0.8)
    plt.title('AAPL Stock Price with Technical Indicators')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('aapl_stock_plot.png')
    print("\nPlot has been saved as 'aapl_stock_plot.png'")
    
    # Save to CSV
    df.to_csv('aapl_data.csv')
    print("\nData has been saved to 'aapl_data.csv'")
    
except ValueError as ve:
    print(f"Data validation error: {str(ve)}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("Please check your internet connection and try again.")

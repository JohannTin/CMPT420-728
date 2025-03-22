from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import time
import matplotlib.pyplot as plt

try:
    # Initialize the TimeSeries class with your API key
    # Note: You'll need to get a free API key from https://www.alphavantage.co/support/#api-key
    ts = TimeSeries(key='Z3WPH5FVFZ3EW8Y1')
    
    # Get daily data for AAPL
    data, meta_data = ts.get_daily(symbol='AAPL', outputsize='compact')
    
    # Convert to pandas DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')
    df.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Convert index to datetime
    df.index = pd.to_datetime(df.index)
    
    # Convert string values to float
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Sort by date
    df = df.sort_index()
    
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
    
    # Create a plot of closing prices
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Closing Price')
    plt.title('AAPL Stock Closing Prices')
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

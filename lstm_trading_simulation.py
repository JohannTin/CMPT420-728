import pandas as pd
import numpy as np

def simulate_trading(predictions_file, initial_bankroll=10000, trade_percentage=0.1):
    # Load predictions
    df = pd.read_csv(predictions_file)
    
    # Initialize variables
    bankroll = initial_bankroll
    position = 0  # 0 = no position, 1 = long position
    shares = 0
    trades = []
    entry_price = 0  # Track entry price for calculating profit/loss
    profitable_trades = 0
    total_trades = 0
    
    # Iterate through predictions
    for i in range(len(df)):
        current_price = df['Actual'].iloc[i]
        signal = df['Signal'].iloc[i]
        date = df['Date'].iloc[i]
        
        # Calculate trade amount (10% of current bankroll)
        trade_amount = bankroll * trade_percentage
        
        # Process buy signal
        if signal == 1 and position == 0:
            shares = trade_amount / current_price
            bankroll -= trade_amount
            position = 1
            entry_price = current_price
            trades.append({
                'Date': date,
                'Action': 'BUY',
                'Price': current_price,
                'Shares': shares,
                'Trade Amount': trade_amount,
                'Bankroll': bankroll
            })
        
        # Process sell signal
        elif signal == -1 and position == 1:
            trade_value = shares * current_price
            bankroll += trade_value
            position = 0
            
            # Calculate if trade was profitable
            profit = trade_value - (shares * entry_price)
            if profit > 0:
                profitable_trades += 1
            total_trades += 1
            
            trades.append({
                'Date': date,
                'Action': 'SELL',
                'Price': current_price,
                'Shares': shares,
                'Trade Amount': trade_value,
                'Bankroll': bankroll,
                'Profit': profit
            })
            shares = 0
    
    # Close any remaining position using the last price
    if position == 1:
        trade_value = shares * df['Actual'].iloc[-1]
        bankroll += trade_value
        
        # Calculate if final trade was profitable
        profit = trade_value - (shares * entry_price)
        if profit > 0:
            profitable_trades += 1
        total_trades += 1
        
        trades.append({
            'Date': df['Date'].iloc[-1],
            'Action': 'SELL',
            'Price': df['Actual'].iloc[-1],
            'Shares': shares,
            'Trade Amount': trade_value,
            'Bankroll': bankroll,
            'Profit': profit
        })
    
    # Calculate performance metrics
    total_profit_loss = bankroll - initial_bankroll
    roi_percentage = (total_profit_loss / initial_bankroll) * 100
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Create trades DataFrame
    trades_df = pd.DataFrame(trades)
    
    return {
        'final_bankroll': bankroll,
        'total_profit_loss': total_profit_loss,
        'roi_percentage': roi_percentage,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'trades': trades_df
    }

def main():
    # Run simulation
    results = simulate_trading('lstm_predictions.csv')
    
    # Print results
    print("\nTrading Simulation Results")
    print("=" * 50)
    print(f"Initial Bankroll: ${10000:,.2f}")
    print(f"Final Bankroll: ${results['final_bankroll']:,.2f}")
    print(f"Total Profit/Loss: ${results['total_profit_loss']:,.2f}")
    print(f"ROI: {results['roi_percentage']:.2f}%")
    print(f"Win Rate: {results['win_rate']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Profitable Trades: {results['profitable_trades']}")
    
    # Print trade history
    print("\nTrade History:")
    print("=" * 50)
    if len(results['trades']) > 0:
        print(results['trades'].to_string(index=False))
    else:
        print("No trades were executed")

if __name__ == "__main__":
    main()
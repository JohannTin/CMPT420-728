import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

# Trading simulation configuration
INITIAL_BANKROLL = 10000
TRADE_PERCENTAGE = 0.3  # Default trade percentage if not using confidence-based sizing
CONFIDENCE_THRESHOLD = 0.6  # Adjusted to match the new model
STOP_LOSS_PERCENTAGE = 0.03  # % stop loss
PREDICTIONS_FILE = 'tft_predictions_improved.csv'  # Updated to use improved predictions

# Confidence-based trade sizing configuration
CONFIDENCE_TRADE_SIZES = {
    0.85: 0.4,  # 40% of bankroll for very high confidence trades
    0.75: 0.3,  # 30% of bankroll for high confidence trades
    0.65: 0.2,  # 20% of bankroll for moderate confidence trades
}

# Kelly Criterion configuration
USE_KELLY = True  # Set to False to use fixed trade sizing
LOOKBACK_PERIOD = 10  # Number of trades to look back for calculating dynamic win rate
AVERAGE_WIN_RETURN = 0.02  # Average return on winning trades (2%)
AVERAGE_LOSS_RETURN = 0.01  # Average return on losing trades (1%)
MAX_KELLY_FRACTION = 0.05  # Maximum fraction of bankroll to risk (conservative approach)
MIN_TRADES_FOR_KELLY = 5  # Minimum number of trades before using Kelly sizing

def calculate_kelly_fraction(recent_trades):
    """Calculate the optimal Kelly fraction for position sizing using dynamic win rate"""
    if len(recent_trades) < MIN_TRADES_FOR_KELLY:
        return MAX_KELLY_FRACTION * 0.5  # Use conservative sizing when insufficient trade history
    
    # Calculate dynamic win rate from recent trades
    win_rate = sum(1 for trade in recent_trades if trade > 0) / len(recent_trades)
    
    # Kelly Formula: K = (p*b - q)/b where b = win_amount/loss_amount
    b = AVERAGE_WIN_RETURN/AVERAGE_LOSS_RETURN  # odds ratio
    q = 1 - win_rate
    kelly = (win_rate*b - q)/b
    
    # Conservative approach: use half-Kelly or limit to maximum fraction
    kelly = min(kelly/2, MAX_KELLY_FRACTION)
    return max(kelly, 0)  # Ensure non-negative fraction

def simulate_trading(predictions_file=PREDICTIONS_FILE, 
                    initial_bankroll=INITIAL_BANKROLL, 
                    trade_percentage=TRADE_PERCENTAGE, 
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    stop_loss_percentage=STOP_LOSS_PERCENTAGE,
                    use_kelly=USE_KELLY):
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
    stop_loss_triggered = 0  # Track number of stop loss triggers
    
    # Initialize recent trades history for dynamic Kelly calculation
    recent_trade_profits = deque(maxlen=LOOKBACK_PERIOD)
    kelly_fraction = MAX_KELLY_FRACTION * 0.5  # Start conservative
    
    # Create a new column for tracking equity over time
    dates = []
    equity = []
    trade_dates = []
    trade_types = []
    
    # Make sure we have a Close column for simulation
    if 'Close' not in df.columns:
        print("ERROR: No Close price column found in predictions file")
        return None
    
    # Iterate through predictions
    for i in range(len(df)):
        current_price = df['Close'].iloc[i]
        signal = df['Signal'].iloc[i]
        date = df['Date'].iloc[i]
        confidence = df['Confidence'].iloc[i] if 'Confidence' in df.columns else 1.0
        
        # Track equity curve
        dates.append(date)
        equity.append(bankroll)
        
        # Skip if price is NaN
        if pd.isna(current_price):
            continue
        
        # Calculate trade size
        if use_kelly:
            # Recalculate Kelly fraction based on recent trade history
            kelly_fraction = calculate_kelly_fraction(recent_trade_profits)
            
            # Adjust Kelly fraction based on confidence
            confidence_adjusted_kelly = kelly_fraction * (confidence/CONFIDENCE_THRESHOLD)
            trade_amount = bankroll * min(confidence_adjusted_kelly, MAX_KELLY_FRACTION)
        else:
            # Determine trade percentage based on confidence using configuration
            current_trade_percentage = 0  # Default to no trade
            for conf_threshold, trade_size in sorted(CONFIDENCE_TRADE_SIZES.items(), reverse=True):
                if confidence >= conf_threshold:
                    current_trade_percentage = trade_size
                    break
            
            # Calculate trade amount based on confidence-adjusted percentage
            trade_amount = bankroll * current_trade_percentage
        
        # Check stop loss if we have a position
        if position == 1:
            loss_percentage = (current_price - entry_price) / entry_price
            if loss_percentage <= -stop_loss_percentage:
                # Stop loss triggered - sell position
                trade_value = shares * current_price
                bankroll += trade_value
                position = 0
                
                # Calculate loss and update trade history
                profit = trade_value - (shares * entry_price)
                recent_trade_profits.append(profit)
                total_trades += 1
                stop_loss_triggered += 1
                
                trades.append({
                    'Date': date,
                    'Action': 'STOP_LOSS',
                    'Price': current_price,
                    'Shares': shares,
                    'Trade Amount': trade_value,
                    'Bankroll': bankroll,
                    'Profit': profit,
                    'Confidence': confidence,
                    'Loss Percentage': loss_percentage * 100,
                    'Kelly Fraction': kelly_fraction if use_kelly else None
                })
                
                # Record trade for plotting
                trade_dates.append(date)
                trade_types.append('STOP_LOSS')
                
                shares = 0
                continue
        
        # Process buy signal with confidence check
        if signal == 1 and position == 0 and confidence >= confidence_threshold:
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
                'Bankroll': bankroll,
                'Confidence': confidence,
                'Kelly Fraction': kelly_fraction if use_kelly else None
            })
            
            # Record trade for plotting
            trade_dates.append(date)
            trade_types.append('BUY')
        
        # Process sell signal
        elif signal == -1 and position == 1:
            trade_value = shares * current_price
            bankroll += trade_value
            position = 0
            
            # Calculate if trade was profitable and update history
            profit = trade_value - (shares * entry_price)
            recent_trade_profits.append(profit)
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
                'Profit': profit,
                'Confidence': confidence,
                'Kelly Fraction': kelly_fraction if use_kelly else None
            })
            
            # Record trade for plotting
            trade_dates.append(date)
            trade_types.append('SELL')
            
            shares = 0
    
    # Close any remaining position using the last price
    if position == 1:
        trade_value = shares * df['Close'].iloc[-1]
        bankroll += trade_value
        
        # Calculate if final trade was profitable
        profit = trade_value - (shares * entry_price)
        recent_trade_profits.append(profit)
        if profit > 0:
            profitable_trades += 1
        total_trades += 1
        
        trades.append({
            'Date': df['Date'].iloc[-1],
            'Action': 'SELL',
            'Price': df['Close'].iloc[-1],
            'Shares': shares,
            'Trade Amount': trade_value,
            'Bankroll': bankroll,
            'Profit': profit,
            'Confidence': df['Confidence'].iloc[-1] if 'Confidence' in df.columns else 1.0,
            'Kelly Fraction': kelly_fraction if use_kelly else None
        })
        
        # Record final trade for plotting
        trade_dates.append(df['Date'].iloc[-1])
        trade_types.append('SELL')
    
    # Calculate performance metrics
    total_profit_loss = bankroll - initial_bankroll
    roi_percentage = (total_profit_loss / initial_bankroll) * 100
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Create trades DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Calculate final Kelly metrics if using Kelly
    final_kelly = calculate_kelly_fraction(recent_trade_profits) if use_kelly else None
    recent_win_rate = (sum(1 for trade in recent_trade_profits if trade > 0) / len(recent_trade_profits)) if recent_trade_profits and use_kelly else 0
    
    # Plot equity curve and trades
    plt.figure(figsize=(15, 8))
    
    # Equity curve
    plt.plot(dates, equity, label='Equity', color='blue')
    
    # Annotate buy/sell points
    buy_points = [date for date, action in zip(trade_dates, trade_types) if action == 'BUY']
    sell_points = [date for date, action in zip(trade_dates, trade_types) if action == 'SELL']
    stop_loss_points = [date for date, action in zip(trade_dates, trade_types) if action == 'STOP_LOSS']
    
    # Find equity values at trade points
    buy_y = [equity[dates.index(date)] if date in dates else None for date in buy_points]
    sell_y = [equity[dates.index(date)] if date in dates else None for date in sell_points]
    stop_loss_y = [equity[dates.index(date)] if date in dates else None for date in stop_loss_points]
    
    # Filter out None values
    buy_points = [x for i, x in enumerate(buy_points) if buy_y[i] is not None]
    buy_y = [y for y in buy_y if y is not None]
    sell_points = [x for i, x in enumerate(sell_points) if sell_y[i] is not None]
    sell_y = [y for y in sell_y if y is not None]
    stop_loss_points = [x for i, x in enumerate(stop_loss_points) if stop_loss_y[i] is not None]
    stop_loss_y = [y for y in stop_loss_y if y is not None]
    
    # Plot trade points
    plt.scatter(buy_points, buy_y, color='green', marker='^', s=100, label='Buy')
    plt.scatter(sell_points, sell_y, color='blue', marker='v', s=100, label='Sell')
    plt.scatter(stop_loss_points, stop_loss_y, color='red', marker='x', s=100, label='Stop Loss')
    
    # Add horizontal line for initial bankroll
    plt.axhline(y=initial_bankroll, color='gray', linestyle='--', label=f'Initial ${initial_bankroll}')
    
    plt.title(f'Equity Curve - {"Kelly" if use_kelly else "Fixed"} Sizing')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Format x-axis dates
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Save plot
    plt.savefig(f'equity_curve_{"kelly" if use_kelly else "fixed"}.png')
    
    return {
        'final_bankroll': bankroll,
        'total_profit_loss': total_profit_loss,
        'roi_percentage': roi_percentage,
        'win_rate': win_rate,
        'total_trades': total_trades,
        'profitable_trades': profitable_trades,
        'stop_loss_triggered': stop_loss_triggered,
        'trades': trades_df,
        'final_kelly_fraction': final_kelly,
        'recent_win_rate': recent_win_rate,
        'equity_curve': pd.DataFrame({'Date': dates, 'Equity': equity})
    }

def main():
    print("\nRunning TFT Trading Simulation with Kelly Criterion...")
    kelly_results = simulate_trading(use_kelly=True)
    
    print("\nRunning TFT Trading Simulation with Fixed Sizing...")
    fixed_results = simulate_trading(use_kelly=False)
    
    if kelly_results is None or fixed_results is None:
        print("Error in simulation. Check your predictions file.")
        return
    
    # Print results for Kelly Criterion
    print("\nTrading Simulation Results (Kelly Criterion)")
    print("=" * 50)
    print(f"Initial Bankroll: ${INITIAL_BANKROLL:,.2f}")
    print(f"Lookback Period: {LOOKBACK_PERIOD} trades")
    print(f"Final Kelly Fraction: {kelly_results['final_kelly_fraction']:.1%}")
    print(f"Recent Win Rate: {kelly_results['recent_win_rate']:.1%}")
    print(f"Max Position Size: {MAX_KELLY_FRACTION:.1%}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD:.2f}")
    print(f"Stop Loss Percentage: {STOP_LOSS_PERCENTAGE:.1%}")
    print(f"Final Bankroll: ${kelly_results['final_bankroll']:,.2f}")
    print(f"Total Profit/Loss: ${kelly_results['total_profit_loss']:,.2f}")
    print(f"ROI: {kelly_results['roi_percentage']:.2f}%")
    print(f"Overall Win Rate: {kelly_results['win_rate']:.2f}%")
    print(f"Total Trades: {kelly_results['total_trades']}")
    print(f"Profitable Trades: {kelly_results['profitable_trades']}")
    print(f"Stop Loss Triggers: {kelly_results['stop_loss_triggered']}")
    
    # Print results for Fixed Sizing
    print("\nTrading Simulation Results (Fixed Sizing)")
    print("=" * 50)
    print(f"Initial Bankroll: ${INITIAL_BANKROLL:,.2f}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD:.2f}")
    print(f"Stop Loss Percentage: {STOP_LOSS_PERCENTAGE:.1%}")
    print(f"Final Bankroll: ${fixed_results['final_bankroll']:,.2f}")
    print(f"Total Profit/Loss: ${fixed_results['total_profit_loss']:,.2f}")
    print(f"ROI: {fixed_results['roi_percentage']:.2f}%")
    print(f"Win Rate: {fixed_results['win_rate']:.2f}%")
    print(f"Total Trades: {fixed_results['total_trades']}")
    print(f"Profitable Trades: {fixed_results['profitable_trades']}")
    print(f"Stop Loss Triggers: {fixed_results['stop_loss_triggered']}")
    
    # Compare strategies
    print("\nPerformance Comparison")
    print("=" * 50)
    kelly_roi = kelly_results['roi_percentage']
    fixed_roi = fixed_results['roi_percentage']
    
    if kelly_roi > fixed_roi:
        print(f"Kelly Criterion outperformed Fixed Sizing by {kelly_roi - fixed_roi:.2f}% ROI")
    elif fixed_roi > kelly_roi:
        print(f"Fixed Sizing outperformed Kelly Criterion by {fixed_roi - kelly_roi:.2f}% ROI")
    else:
        print("Both methods performed equally")
    
    # Plot combined equity curves
    plt.figure(figsize=(15, 8))
    
    kelly_equity = kelly_results['equity_curve']
    fixed_equity = fixed_results['equity_curve']
    
    plt.plot(kelly_equity['Date'], kelly_equity['Equity'], label='Kelly Criterion', color='blue')
    plt.plot(fixed_equity['Date'], fixed_equity['Equity'], label='Fixed Sizing', color='green')
    plt.axhline(y=INITIAL_BANKROLL, color='gray', linestyle='--', label=f'Initial ${INITIAL_BANKROLL}')
    
    plt.title('Equity Curve Comparison')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('equity_comparison.png')
    
    # Print a sample of recent trades
    print(f"\nRecent Trades Sample ({min(20, len(kelly_results['trades']))} of {len(kelly_results['trades'])} trades):")
    print("=" * 50)
    if len(kelly_results['trades']) > 0:
        pd.set_option('display.max_rows', 10)
        pd.set_option('display.width', 1000)
        print(kelly_results['trades'].tail(20).to_string(index=False))
        pd.reset_option('display.max_rows')
        pd.reset_option('display.width')
    else:
        print("No trades were executed")
    
    # Save trade history to file
    kelly_results['trades'].to_csv('tft_trades_kelly_improved.csv', index=False)
    fixed_results['trades'].to_csv('tft_trades_fixed_improved.csv', index=False)
    print("\nTrade histories saved to tft_trades_kelly_improved.csv and tft_trades_fixed_improved.csv")
    print("Equity curve plots saved to equity_curve_kelly.png, equity_curve_fixed.png and equity_comparison.png")

if __name__ == "__main__":
    main()
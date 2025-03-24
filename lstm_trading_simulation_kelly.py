import pandas as pd
import numpy as np
from collections import deque

# Trading simulation configuration
INITIAL_BANKROLL = 10000
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence level required to execute trades
STOP_LOSS_PERCENTAGE = 0.05  # % stop loss
PREDICTIONS_FILE = 'lstm_predictions.csv'

# Kelly Criterion configuration
LOOKBACK_PERIOD = 8  # Number of trades to look back for calculating dynamic win rate
AVERAGE_WIN_RETURN = 0.02  # Average return on winning trades (2%)
AVERAGE_LOSS_RETURN = 0.01  # Average return on losing trades (1%)
MAX_KELLY_FRACTION = 0.5  # Maximum fraction of bankroll to risk (conservative approach)
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
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    stop_loss_percentage=STOP_LOSS_PERCENTAGE):
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
    
    # Iterate through predictions
    for i in range(len(df)):
        current_price = df['Actual'].iloc[i]
        signal = df['Signal'].iloc[i]
        date = df['Date'].iloc[i]
        confidence = df['Confidence'].iloc[i] if 'Confidence' in df.columns else 1.0
        
        # Recalculate Kelly fraction based on recent trade history
        kelly_fraction = calculate_kelly_fraction(recent_trade_profits)
        
        # Adjust Kelly fraction based on confidence
        confidence_adjusted_kelly = kelly_fraction * (confidence/CONFIDENCE_THRESHOLD)
        trade_amount = bankroll * confidence_adjusted_kelly
        
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
                    'Kelly Fraction': kelly_fraction
                })
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
                'Confidence': confidence
            })
        
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
                'Kelly Fraction': kelly_fraction
            })
            shares = 0
    
    # Close any remaining position using the last price
    if position == 1:
        trade_value = shares * df['Actual'].iloc[-1]
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
            'Price': df['Actual'].iloc[-1],
            'Shares': shares,
            'Trade Amount': trade_value,
            'Bankroll': bankroll,
            'Profit': profit,
            'Confidence': df['Confidence'].iloc[-1] if 'Confidence' in df.columns else 1.0,
            'Kelly Fraction': kelly_fraction
        })
    
    # Calculate performance metrics
    total_profit_loss = bankroll - initial_bankroll
    roi_percentage = (total_profit_loss / initial_bankroll) * 100
    win_rate = (profitable_trades / total_trades * 100) if total_trades > 0 else 0
    
    # Create trades DataFrame
    trades_df = pd.DataFrame(trades)
    
    # Calculate final Kelly metrics
    final_kelly = calculate_kelly_fraction(recent_trade_profits)
    
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
        'recent_win_rate': (sum(1 for trade in recent_trade_profits if trade > 0) / len(recent_trade_profits)) if recent_trade_profits else 0
    }

def main():
    # Run simulation
    results = simulate_trading()
    
    # Print results
    print("\nTrading Simulation Results (Kelly Criterion)")
    print("=" * 50)
    print(f"Initial Bankroll: ${INITIAL_BANKROLL:,.2f}")
    print(f"Lookback Period: {LOOKBACK_PERIOD} trades")
    print(f"Final Kelly Fraction: {results['final_kelly_fraction']:.1%}")
    print(f"Recent Win Rate: {results['recent_win_rate']:.1%}")
    print(f"Max Position Size: {MAX_KELLY_FRACTION:.1%}")
    print(f"Confidence Threshold: {CONFIDENCE_THRESHOLD:.2f}")
    print(f"Stop Loss Percentage: {STOP_LOSS_PERCENTAGE:.1%}")
    print(f"Final Bankroll: ${results['final_bankroll']:,.2f}")
    print(f"Total Profit/Loss: ${results['total_profit_loss']:,.2f}")
    print(f"ROI: {results['roi_percentage']:.2f}%")
    print(f"Overall Win Rate: {results['win_rate']:.2f}%")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Profitable Trades: {results['profitable_trades']}")
    print(f"Stop Loss Triggers: {results['stop_loss_triggered']}")
    
    # Print trade history
    print("\nTrade History:")
    print("=" * 50)
    if len(results['trades']) > 0:
        print(results['trades'].to_string(index=False))
    else:
        print("No trades were executed")

if __name__ == "__main__":
    main()
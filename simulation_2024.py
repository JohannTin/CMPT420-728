import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Trading simulation configuration
INITIAL_BANKROLL = 10000
TRADE_PERCENTAGE = 0.3  # Default trade percentage if not using confidence-based sizing
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence level required to execute trades
STOP_LOSS_PERCENTAGE = 0.03  # % stop loss

# Model prediction files
MODEL_FILES = {
    'LSTM': 'lstm_predictions.csv',
    'TCN': 'Price_History_Model/TCN_predictions.csv',
    'TFT': 'tft_predictions.csv'
}

# Confidence-based trade sizing configuration
CONFIDENCE_TRADE_SIZES = {
    0.99: 0.4,  # 40% of bankroll for very high confidence trades
    0.97: 0.3,  # 30% of bankroll for high confidence trades
    0.95: 0.2,  # 20% of bankroll for moderate confidence trades
    0.90: 0.15, # 15% of bankroll for lower confidence trades
    0.85: 0.1,  # 10% of bankroll for very low confidence trades
    0.70: 0.08, # 8% of bankroll for low confidence trades
    0.60: 0.05, # 5% of bankroll for very low confidence trades
    0.50: 0.03, # 3% of bankroll for extremely low confidence trades
}

def simulate_trading(predictions_file, 
                    initial_bankroll=INITIAL_BANKROLL, 
                    trade_percentage=TRADE_PERCENTAGE, 
                    confidence_threshold=CONFIDENCE_THRESHOLD,
                    stop_loss_percentage=STOP_LOSS_PERCENTAGE):
    # Load predictions
    df = pd.read_csv(predictions_file)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter for 2024 data
    start_date = '2024-01-01'
    end_date = '2024-12-31'
    df = df[(df['Date'] >= start_date) & (df['Date'] <= end_date)]
    
    # Initialize variables
    bankroll = initial_bankroll
    position = 0  # 0 = no position, 1 = long position
    shares = 0
    trades = []
    entry_price = 0  # Track entry price for calculating profit/loss
    profitable_trades = 0
    total_trades = 0
    stop_loss_triggered = 0  # Track number of stop loss triggers
    
    # Iterate through predictions
    for i in range(len(df)):
        current_price = df['Actual'].iloc[i] if 'Actual' in df.columns else df['Actual_Close'].iloc[i]
        signal = df['Signal'].iloc[i]
        date = df['Date'].iloc[i]
        confidence = df['Confidence'].iloc[i] if 'Confidence' in df.columns else 1.0
        
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
                
                # Calculate loss
                profit = trade_value - (shares * entry_price)
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
                    'Loss Percentage': loss_percentage * 100
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
        
        # Process sell signal with confidence check
        elif signal == -1 and position == 1 and confidence >= confidence_threshold:
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
                'Profit': profit,
                'Confidence': confidence
            })
            shares = 0
    
    # Close any remaining position using the last price
    if position == 1:
        trade_value = shares * df['Actual'].iloc[-1] if 'Actual' in df.columns else df['Actual_Close'].iloc[-1]
        bankroll += trade_value
        
        # Calculate if final trade was profitable
        profit = trade_value - (shares * entry_price)
        if profit > 0:
            profitable_trades += 1
        total_trades += 1
        
        trades.append({
            'Date': df['Date'].iloc[-1],
            'Action': 'SELL',
            'Price': df['Actual'].iloc[-1] if 'Actual' in df.columns else df['Actual_Close'].iloc[-1],
            'Shares': shares,
            'Trade Amount': trade_value,
            'Bankroll': bankroll,
            'Profit': profit,
            'Confidence': df['Confidence'].iloc[-1] if 'Confidence' in df.columns else 1.0
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
        'stop_loss_triggered': stop_loss_triggered,
        'trades': trades_df
    }

def plot_performance_comparison(results):
    # Create comparison DataFrame
    comparison_data = []
    for model, result in results.items():
        comparison_data.append({
            'Model': model,
            'ROI': result['roi_percentage'],
            'Win Rate': result['win_rate'],
            'Total Trades': result['total_trades'],
            'Stop Loss Triggers': result['stop_loss_triggered']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Create plots
    plt.figure(figsize=(15, 10))
    
    # ROI Comparison
    plt.subplot(2, 2, 1)
    sns.barplot(x='Model', y='ROI', data=comparison_df)
    plt.title('ROI Comparison (2024)')
    plt.ylabel('ROI (%)')
    
    # Win Rate Comparison
    plt.subplot(2, 2, 2)
    sns.barplot(x='Model', y='Win Rate', data=comparison_df)
    plt.title('Win Rate Comparison (2024)')
    plt.ylabel('Win Rate (%)')
    
    # Total Trades Comparison
    plt.subplot(2, 2, 3)
    sns.barplot(x='Model', y='Total Trades', data=comparison_df)
    plt.title('Total Trades Comparison (2024)')
    plt.ylabel('Number of Trades')
    
    # Stop Loss Triggers Comparison
    plt.subplot(2, 2, 4)
    sns.barplot(x='Model', y='Stop Loss Triggers', data=comparison_df)
    plt.title('Stop Loss Triggers Comparison (2024)')
    plt.ylabel('Number of Stop Loss Triggers')
    
    plt.tight_layout()
    plt.savefig('trading_performance_comparison_2024.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    # Run simulation for all models
    results = {}
    for model_name, file_path in MODEL_FILES.items():
        print(f"\nRunning simulation for {model_name} model...")
        results[model_name] = simulate_trading(file_path)
        
        # Print results for each model
        print(f"\n{model_name} Trading Simulation Results (2024)")
        print("=" * 50)
        print(f"Initial Bankroll: ${INITIAL_BANKROLL:,.2f}")
        print(f"Final Bankroll: ${results[model_name]['final_bankroll']:,.2f}")
        print(f"Total Profit/Loss: ${results[model_name]['total_profit_loss']:,.2f}")
        print(f"ROI: {results[model_name]['roi_percentage']:.2f}%")
        print(f"Win Rate: {results[model_name]['win_rate']:.2f}%")
        print(f"Total Trades: {results[model_name]['total_trades']}")
        print(f"Profitable Trades: {results[model_name]['profitable_trades']}")
        print(f"Stop Loss Triggers: {results[model_name]['stop_loss_triggered']}")
    
    # Create performance comparison plots
    plot_performance_comparison(results)

if __name__ == "__main__":
    main() 
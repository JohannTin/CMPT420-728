import numpy as np
import pandas as pd
from itertools import product
from lstm_trading_simulation_kelly import simulate_trading
import json
from datetime import datetime

# Define parameter search spaces
param_grid = {
    "confidence_threshold": np.linspace(0.7, 0.9, 5),        # [0.70, 0.75, 0.80, 0.85, 0.90]
    "stop_loss_percentage": np.linspace(0.03, 0.06, 4),      # [0.03, 0.04, 0.05, 0.06]
    "lookback_period": [8, 10, 12, 15, 20],                  # Explore near best (10)
    "max_kelly_fraction": np.linspace(0.05, 0.3, 6)          # [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
}

def grid_search_optimization():
    """
    Perform grid search to find optimal parameters for trading simulation
    """
    # Generate all possible combinations of parameters
    param_combinations = [dict(zip(param_grid.keys(), v)) for v in product(*param_grid.values())]
    
    # Store results
    results = []
    best_roi = float('-inf')
    best_params = None
    total_combinations = len(param_combinations)
    
    print(f"Starting grid search with {total_combinations} parameter combinations...")
    
    for i, params in enumerate(param_combinations, 1):
        # Run simulation with current parameter set
        sim_results = simulate_trading(
            confidence_threshold=params['confidence_threshold'],
            stop_loss_percentage=params['stop_loss_percentage'],
            initial_bankroll=10000  # Keep initial bankroll constant
        )
        
        # Store results
        result = {
            'params': params,
            'roi': sim_results['roi_percentage'],
            'total_trades': sim_results['total_trades'],
            'win_rate': sim_results['win_rate'],
            'final_bankroll': sim_results['final_bankroll']
        }
        results.append(result)
        
        # Update best parameters if current ROI is better
        if sim_results['roi_percentage'] > best_roi:
            best_roi = sim_results['roi_percentage']
            best_params = params.copy()
        
        # Print progress
        if i % 10 == 0:
            print(f"Progress: {i}/{total_combinations} combinations tested")
            print(f"Current best ROI: {best_roi:.2f}% with params: {best_params}")
    
    # Convert results to DataFrame for analysis
    results_df = pd.DataFrame(results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f'optimization_results_{timestamp}.csv', index=False)
    
    # Save best parameters
    with open(f'best_params_{timestamp}.json', 'w') as f:
        json.dump(best_params, f, indent=4)
    
    return best_params, results_df

def print_optimization_results(best_params, results_df):
    """
    Print detailed optimization results
    """
    print("\nOptimization Results")
    print("=" * 50)
    print("\nBest Parameters Found:")
    for param, value in best_params.items():
        print(f"{param}: {value:.4f}")
    
    print("\nPerformance Statistics:")
    print(f"Best ROI: {results_df['roi'].max():.2f}%")
    print(f"Average ROI: {results_df['roi'].mean():.2f}%")
    print(f"ROI Standard Deviation: {results_df['roi'].std():.2f}%")
    print(f"Total parameter combinations tested: {len(results_df)}")
    
    # Print parameter importance analysis
    print("\nParameter Correlation with ROI:")
    correlations = {}
    for param in best_params.keys():
        correlation = results_df['roi'].corr(results_df['params'].apply(lambda x: x[param]))
        correlations[param] = correlation
        print(f"{param}: {correlation:.4f}")

def main():
    print("Starting trading parameter optimization...")
    best_params, results_df = grid_search_optimization()
    print_optimization_results(best_params, results_df)

if __name__ == "__main__":
    main() 
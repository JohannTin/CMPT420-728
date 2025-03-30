import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
import os
import time
from datetime import datetime

# Configuration
CONFIG = {
    'SYMBOL': 'AAPL',
    'SEQUENCE_LENGTH': 30,
    'FORECAST_HORIZON': 5,  # Predict 5 days ahead
    'TRAIN_SIZE_RATIO': 0.80,
    'EPOCHS': 100,
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 0.0005,
    'HIDDEN_SIZE': 128,  # Increased from 64
    'NUM_HEADS': 8,      # Increased from 4
    'NUM_QUANTILES': 3,  # for P10, P50, P90
    'DROPOUT_RATE': 0.1, # Reduced dropout
    'CONFIDENCE_THRESHOLD': 0.6,
    'WEIGHT_DECAY': 1e-5,  # L2 regularization
    'GRADIENT_CLIP': 1.0,  # Clip gradients
    'START_DATE': '2015-01-01',  # More training data
    'END_DATE': '2025-01-01'
}

# Ensure deterministic behavior
torch.manual_seed(42)
np.random.seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting with TFT."""
    
    def __init__(self, data, static_covariates, seq_length, target_idx=0, forecast_horizon=1):
        self.data = data
        self.static_covariates = static_covariates
        self.seq_length = seq_length
        self.target_idx = target_idx
        self.forecast_horizon = forecast_horizon
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.forecast_horizon + 1
    
    def __getitem__(self, idx):
        # Get sequence of data
        past_data = self.data[idx:idx + self.seq_length]
        
        # Target is n days ahead
        future_idx = idx + self.seq_length + self.forecast_horizon - 1
        target = self.data[future_idx, self.target_idx]
        
        # Static covariates are the same for each sequence
        static = self.static_covariates.copy()
        
        return {
            'past_data': torch.tensor(past_data, dtype=torch.float32),
            'static': torch.tensor(static, dtype=torch.float32),
            'target': torch.tensor(target, dtype=torch.float32)
        }

class GatedResidualNetwork(nn.Module):
    """Gated Residual Network as described in the TFT paper."""
    
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.1, context_size=None):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size
        
        # Layer 1
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Context projection if context is provided
        if context_size is not None:
            self.context_projection = nn.Linear(context_size, hidden_size, bias=False)
        
        # Layer 2
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # GLU
        self.glu_layer = nn.Linear(hidden_size, output_size * 2)
        
        # Residual connection
        if input_size != output_size:
            self.skip_layer = nn.Linear(input_size, output_size)
        else:
            self.skip_layer = None
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x, context=None):
        # Main branch
        main = self.fc1(x)
        
        # Add context if available
        if self.context_size is not None and context is not None:
            main = main + self.context_projection(context)
        
        # Nonlinear activation
        main = F.elu(main)
        
        # Second linear layer
        main = self.fc2(main)
        
        # Apply dropout
        main = self.dropout(main)
        
        # GLU mechanism
        glu_input = self.glu_layer(F.elu(self.fc1(x)))
        glu_output, glu_gate = torch.chunk(glu_input, 2, dim=-1)
        glu_output = glu_output * torch.sigmoid(glu_gate)
        
        # Add residual connection
        if self.skip_layer is not None:
            skip = self.skip_layer(x)
        else:
            skip = x
        
        # Combine with residual connection
        output = self.layer_norm(skip + glu_output)
        
        return output

class VariableSelectionNetwork(nn.Module):
    """Variable selection network from TFT."""
    
    def __init__(self, input_sizes, hidden_size, output_size, dropout_rate=0.1, context_size=None):
        super(VariableSelectionNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.num_inputs = len(input_sizes)
        
        # GRN for variable weights
        self.weight_grn_input_size = sum(input_sizes)
        if context_size is not None:
            self.weight_grn_input_size += context_size
        
        self.weight_grn = GatedResidualNetwork(
            input_size=self.weight_grn_input_size,
            hidden_size=hidden_size,
            output_size=self.num_inputs,
            dropout_rate=dropout_rate
        )
        
        # GRN for each variable
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_size=input_size,
                hidden_size=hidden_size,
                output_size=output_size,
                dropout_rate=dropout_rate
            ) for input_size in input_sizes
        ])
    
    def forward(self, inputs, context=None):
        # Process each variable with its own GRN
        var_outputs = [grn(var_input) for grn, var_input in zip(self.variable_grns, inputs)]
        
        # Create variable weights
        flat_inputs = torch.cat([inp.unsqueeze(-2) for inp in inputs], dim=-2)
        flat_inputs = flat_inputs.view(flat_inputs.size(0), -1)
        
        # Add context if provided
        if context is not None:
            weight_input = torch.cat([flat_inputs, context], dim=-1)
        else:
            weight_input = flat_inputs
        
        # Generate weights using the weight GRN
        weights = self.weight_grn(weight_input)
        weights = F.softmax(weights, dim=-1)
        
        # Weight the variable outputs
        combined_output = torch.zeros_like(var_outputs[0])
        var_weights = []
        
        for i, output in enumerate(var_outputs):
            # Extract the weight for this variable
            var_weight = weights[..., i].unsqueeze(-1)
            var_weights.append(var_weight.detach().cpu().numpy().mean())
            # Multiply the variable output by its weight
            combined_output = combined_output + var_weight * output
        
        return combined_output, var_weights

class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-head Attention from TFT."""
    
    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        
        # Create key and query projection layers for each head
        self.key_projections = nn.ModuleList(
            [nn.Linear(hidden_size, self.head_size) for _ in range(num_heads)]
        )
        self.query_projections = nn.ModuleList(
            [nn.Linear(hidden_size, self.head_size) for _ in range(num_heads)]
        )
        
        # Shared value projection
        self.value_projection = nn.Linear(hidden_size, self.head_size)
        
        # Output projection
        self.output_projection = nn.Linear(self.head_size, hidden_size)
        
        # Dropout
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, queries, keys, values, mask=None):
        batch_size = queries.shape[0]
        seq_len = queries.shape[1]
        
        # Project values (shared across all heads)
        projected_values = self.value_projection(values)
        
        # Initialize outputs
        outputs = torch.zeros(batch_size, seq_len, self.head_size, device=queries.device)
        attention_weights = torch.zeros(batch_size, self.num_heads, seq_len, keys.shape[1], device=queries.device)
        
        # Process each head
        for h in range(self.num_heads):
            # Project queries and keys for this head
            projected_queries = self.query_projections[h](queries)
            projected_keys = self.key_projections[h](keys)
            
            # Calculate attention scores
            scores = torch.matmul(projected_queries, projected_keys.transpose(-2, -1))
            scores = scores / (self.head_size ** 0.5)
            
            # Apply mask if provided
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)
            
            # Apply softmax
            attention = F.softmax(scores, dim=-1)
            attention = self.dropout(attention)
            
            # Save attention weights for interpretability
            attention_weights[:, h] = attention
            
            # Apply attention to values
            head_output = torch.matmul(attention, projected_values)
            outputs = outputs + head_output
        
        # Average over all heads
        outputs = outputs / self.num_heads
        
        # Final projection
        outputs = self.output_projection(outputs)
        
        # Return average attention weights across heads for interpretability
        attention_weights = attention_weights.mean(dim=1)
        
        return outputs, attention_weights

class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for time series forecasting."""
    
    def __init__(self, 
                 num_static_features=1,
                 num_known_features=3,
                 num_observed_features=4,
                 hidden_size=128,
                 num_heads=8,
                 dropout_rate=0.1,
                 num_quantiles=3):
        super(TemporalFusionTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_quantiles = num_quantiles
        
        # Static, known, and observed feature counts
        self.num_static_features = num_static_features
        self.num_known_features = num_known_features
        self.num_observed_features = num_observed_features
        
        # Create lists of input sizes - each feature is size 1
        static_input_sizes = [1] * num_static_features
        known_input_sizes = [1] * num_known_features
        observed_input_sizes = [1] * num_observed_features
        
        # Variable selection networks
        self.static_vsn = VariableSelectionNetwork(
            input_sizes=static_input_sizes,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )
        
        self.known_vsn = VariableSelectionNetwork(
            input_sizes=known_input_sizes,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate,
            context_size=hidden_size  # Context from static variables
        ) if num_known_features > 0 else None
        
        self.observed_vsn = VariableSelectionNetwork(
            input_sizes=observed_input_sizes,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate,
            context_size=hidden_size  # Context from static variables
        )
        
        # Static context vectors for various purposes
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )
        
        self.static_context_initial_hidden = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )
        
        self.static_context_initial_cell = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )
        
        self.static_context_enrichment = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )
        
        # Sequence-to-sequence layer (LSTM)
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        
        # Skip connection gate for LSTM outputs
        self.lstm_skip_connection = nn.Linear(hidden_size, hidden_size)
        self.post_lstm_gate = nn.Linear(hidden_size, hidden_size)
        self.post_lstm_norm = nn.LayerNorm(hidden_size)
        
        # Static enrichment layer
        self.static_enrichment = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate,
            context_size=hidden_size
        )
        
        # Temporal attention layer
        self.multihead_attention = InterpretableMultiHeadAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate
        )
        
        # Skip connection gate for attention outputs
        self.attention_skip_connection = nn.Linear(hidden_size, hidden_size)
        self.post_attention_gate = nn.Linear(hidden_size, hidden_size)
        self.post_attention_norm = nn.LayerNorm(hidden_size)
        
        # Position-wise feed-forward layer
        self.position_wise_ff = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate
        )
        
        # Quantile prediction layers
        self.quantile_proj = nn.Linear(hidden_size, num_quantiles)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for better training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x, static=None, mask=None):
        """
        Forward pass of TFT.
        
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            static: Static features of shape [batch_size, num_static_features]
            mask: Optional mask for self-attention
            
        Returns:
            Dictionary containing model outputs
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]
        
        # Process static features if provided
        if static is not None:
            # Split static features
            static_inputs = [static[:, i:i+1] for i in range(static.size(1))]
            
            # Apply variable selection to static features
            static_embedding, static_weights = self.static_vsn(static_inputs)
            
            # Generate context vectors for different parts of the network
            context_var_selection = self.static_context_variable_selection(static_embedding)
            context_init_hidden = self.static_context_initial_hidden(static_embedding)
            context_init_cell = self.static_context_initial_cell(static_embedding)
            context_enrichment = self.static_context_enrichment(static_embedding)
        else:
            # Default context vectors when no static features are provided
            static_embedding = torch.zeros(batch_size, self.hidden_size, device=x.device)
            context_var_selection = torch.zeros(batch_size, self.hidden_size, device=x.device)
            context_init_hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
            context_init_cell = torch.zeros(batch_size, self.hidden_size, device=x.device)
            context_enrichment = torch.zeros(batch_size, self.hidden_size, device=x.device)
            static_weights = None
        
        # Container for processed temporal features
        temporal_features = []
        known_weights_list = []
        observed_weights_list = []
        
        # Process each time step
        for t in range(seq_len):
            # Get features for this time step
            features = x[:, t, :]
            
            # Split into known and observed features
            if self.num_known_features > 0:
                known_inputs = [features[:, i:i+1] for i in range(self.num_known_features)]
                observed_inputs = [features[:, self.num_known_features+i:self.num_known_features+i+1] 
                                for i in range(self.num_observed_features)]
                
                # Process known inputs
                known_embedding, known_weights = self.known_vsn(known_inputs, context_var_selection)
                known_weights_list.append(known_weights)
            else:
                known_embedding = torch.zeros(batch_size, self.hidden_size, device=x.device)
                known_weights = None
            
            # Process observed inputs
            observed_embedding, observed_weights = self.observed_vsn(observed_inputs, context_var_selection)
            observed_weights_list.append(observed_weights)
            
            # Combine embeddings
            combined_embedding = known_embedding + observed_embedding
            temporal_features.append(combined_embedding)
        
        # Stack temporal features
        temporal_features = torch.stack(temporal_features, dim=1)
        
        # Initialize LSTM states with static context
        h0 = context_init_hidden.unsqueeze(0)
        c0 = context_init_cell.unsqueeze(0)
        
        # Pass through encoder LSTM
        lstm_output, (hidden, cell) = self.encoder_lstm(temporal_features, (h0, c0))
        
        # Skip connection around LSTM
        lstm_skip = self.lstm_skip_connection(temporal_features)
        lstm_gate = torch.sigmoid(self.post_lstm_gate(lstm_output))
        lstm_output = self.post_lstm_norm(lstm_skip + lstm_gate * lstm_output)
        
        # Static enrichment
        enriched_output = torch.zeros_like(lstm_output)
        for t in range(seq_len):
            enriched_output[:, t] = self.static_enrichment(lstm_output[:, t], context_enrichment)
        
        # Self-attention mechanism
        # Create causal mask for self-attention
        if mask is None:
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
            mask = (~mask).to(x.device)
        
        # Apply self-attention
        attn_output, attn_weights = self.multihead_attention(
            queries=enriched_output, 
            keys=enriched_output, 
            values=enriched_output,
            mask=mask
        )
        
        # Skip connection around attention
        attn_skip = self.attention_skip_connection(enriched_output)
        attn_gate = torch.sigmoid(self.post_attention_gate(attn_output))
        attn_output = self.post_attention_norm(attn_skip + attn_gate * attn_output)
        
        # Position-wise feed-forward layer
        output = torch.zeros_like(attn_output)
        for t in range(seq_len):
            output[:, t] = self.position_wise_ff(attn_output[:, t])
        
        # Generate quantile predictions for the last timestep
        final_hidden = output[:, -1]
        quantiles = self.quantile_proj(final_hidden)
        
        # Return model outputs
        return {
            'quantiles': quantiles,
            'p10': quantiles[:, 0:1],  # First quantile is p10
            'p50': quantiles[:, 1:2],  # Middle quantile is p50
            'p90': quantiles[:, 2:3],  # Last quantile is p90
            'attention_weights': attn_weights,
            'static_weights': static_weights,
            'known_weights': known_weights_list[-1] if known_weights_list else None,
            'observed_weights': observed_weights_list[-1]
        }

def load_and_prepare_data(verbose=True):
    """Load and prepare both price and sentiment data."""
    if verbose:
        print("Loading data from appl_data_with_indicators.csv...")
    df_price = pd.read_csv('appl_data_with_indicators.csv')
    df_price['Date'] = pd.to_datetime(df_price['Unnamed: 0'])
    df_price.set_index('Date', inplace=True)
    df_price = df_price.sort_index()
    
    if verbose:
        print("Loading data from sentiment_analysis_detailed.csv...")
    df_sentiment = pd.read_csv('sentiment_analysis_detailed.csv')
    # Convert published_at to datetime and extract date more robustly
    df_sentiment['Date'] = pd.to_datetime(df_sentiment['published_at'], utc=True).dt.strftime('%Y-%m-%d')
    df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date'])
    df_sentiment = df_sentiment.groupby('Date').agg({
        'sentiment_positive': 'mean',
        'sentiment_neutral': 'mean',
        'sentiment_negative': 'mean'
    }).reset_index()
    df_sentiment['Date'] = pd.to_datetime(df_sentiment['Date'])
    df_sentiment.set_index('Date', inplace=True)
    
    # Merge price and sentiment data
    df = df_price.join(df_sentiment, how='left')
    
    # Forward fill sentiment scores for days without news
    df[['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']] = \
        df[['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']].ffill()
    
    # If there are still NaN values at the start, backward fill
    df[['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']] = \
        df[['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']].bfill()
    
    # Fill any remaining NaN values with neutral sentiment
    df[['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']] = \
        df[['sentiment_positive', 'sentiment_neutral', 'sentiment_negative']].fillna(1/3)
    
    # Filter date range
    df = df[(df.index >= CONFIG['START_DATE']) & (df.index <= CONFIG['END_DATE'])]
    
    # Add date features (known inputs)
    df['day_of_week'] = df.index.dayofweek / 6.0  # Normalize to [0, 1]
    df['day_of_month'] = (df.index.day - 1) / 30.0  # Normalize to [0, 1]
    df['month'] = (df.index.month - 1) / 11.0  # Normalize to [0, 1]
    df['quarter'] = (df.index.quarter - 1) / 3.0  # Normalize to [0, 1]
    df['year'] = (df.index.year - df.index.year.min()) / max(1, (df.index.year.max() - df.index.year.min()))
    
    # Add trend and momentum features
    df['price_momentum_1d'] = df['Close'].pct_change(1)
    df['price_momentum_5d'] = df['Close'].pct_change(5)
    df['price_momentum_20d'] = df['Close'].pct_change(20)
    df['volume_momentum_5d'] = df['Volume'].pct_change(5)
    
    # Add target: n-day future return
    horizon = CONFIG['FORECAST_HORIZON']
    df[f'future_return_{horizon}d'] = df['Close'].pct_change(horizon).shift(-horizon)
    
    # Add binary target (up/down)
    df[f'direction_{horizon}d'] = np.where(df[f'future_return_{horizon}d'] > 0, 1, 0)
    
    if verbose:
        print(f"DataFrame shape: {df.shape}")
        if verbose > 1:
            print(f"DataFrame columns: {df.columns.tolist()}")
    
    return df

def prepare_data_for_tft(df, verbose=True):
    """Prepare data for TFT model."""
    # Check for duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols and verbose:
        print(f"Warning: Duplicate columns found: {duplicate_cols}")
        # Create a new DataFrame with deduplicated columns
        df = df.loc[:, ~df.columns.duplicated()]
        if verbose:
            print(f"After deduplication, shape: {df.shape}")
    
    # Define feature groups
    # Binary target - 1 if price goes up in n days, 0 if down
    target_col = f'direction_{CONFIG["FORECAST_HORIZON"]}d'
    
    # Known features (calendar features - already normalized)
    known_cols = ['day_of_week', 'day_of_month', 'month', 'quarter', 'year']
    
    # Observed features (past measurements we can observe)
    observed_cols = [
        'Close', 'Volume', 'EMA_8', 'SMA_200', 'RSI',
        'price_momentum_1d', 'price_momentum_5d', 'price_momentum_20d',
        'sentiment_positive', 'sentiment_negative'
    ]
    
    # Verify columns exist
    all_cols = known_cols + observed_cols + [target_col]
    for col in all_cols[:]:  # Use slice to allow modification during iteration
        if col not in df.columns:
            if verbose:
                print(f"Warning: Column {col} not found in data. Removing from features.")
            if col in known_cols:
                known_cols.remove(col)
            if col in observed_cols:
                observed_cols.remove(col)
    
    # Select and clean data
    feature_cols = known_cols + observed_cols
    for col in feature_cols + [target_col]:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df_clean = df[feature_cols + [target_col]].copy()
    df_clean = df_clean.dropna()
    
    if verbose:
        print(f"Clean data shape: {df_clean.shape}")
    
    # Normalize observed features (not the calendar features which are already normalized)
    scaler = StandardScaler()
    
    # Fit scaler only on observed features
    scaler.fit(df_clean[observed_cols])
    
    # Scale observed features
    observed_scaled = scaler.transform(df_clean[observed_cols])
    
    # Keep known features as is (already normalized) and add scaled observed features
    data = np.concatenate([
        df_clean[known_cols].values,
        observed_scaled,
        df_clean[[target_col]].values
    ], axis=1)
    
    # Create static features (simple for now - just 1 for AAPL)
    static_features = [1.0]  # Stock ID
    
    # Define constants for model
    num_known = len(known_cols)
    num_observed = len(observed_cols)
    num_static = len(static_features)
    
    # Save info for later
    CONFIG['target_scaler'] = scaler
    CONFIG['target_idx'] = len(feature_cols)  # Target is the last column
    CONFIG['known_cols'] = known_cols
    CONFIG['observed_cols'] = observed_cols
    
    # Create feature importance dictionary
    feature_importance = {
        'static': ['stock_id'],
        'known': known_cols,
        'observed': observed_cols
    }
    CONFIG['feature_importance'] = feature_importance
    
    return data, static_features, df_clean.index, num_static, num_known, num_observed

def train_model(data, static_features, dates, num_static, num_known, num_observed, verbose=True):
    """Train TFT model."""
    # Split data into train and test
    train_size = int(len(data) * CONFIG['TRAIN_SIZE_RATIO'])
    train_data = data[:train_size]
    test_data = data[train_size:]
    train_dates = dates[:train_size]
    test_dates = dates[train_size:]
    
    if verbose:
        print(f"Train size: {train_data.shape}, Test size: {test_data.shape}")
    
    # Create datasets
    train_dataset = TimeSeriesDataset(
        data=train_data,
        static_covariates=static_features,
        seq_length=CONFIG['SEQUENCE_LENGTH'],
        target_idx=CONFIG['target_idx'],
        forecast_horizon=CONFIG['FORECAST_HORIZON']
    )
    
    test_dataset = TimeSeriesDataset(
        data=test_data,
        static_covariates=static_features,
        seq_length=CONFIG['SEQUENCE_LENGTH'],
        target_idx=CONFIG['target_idx'],
        forecast_horizon=CONFIG['FORECAST_HORIZON']
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=CONFIG['BATCH_SIZE'],
        shuffle=False
    )
    
    # Create TFT model with fixed dimensions
    model = TemporalFusionTransformer(
        num_static_features=num_static,
        num_known_features=num_known,
        num_observed_features=num_observed,
        hidden_size=CONFIG['HIDDEN_SIZE'],
        num_heads=CONFIG['NUM_HEADS'],
        dropout_rate=CONFIG['DROPOUT_RATE'],
        num_quantiles=CONFIG['NUM_QUANTILES']
    ).to(device)
    
    # Define combined loss function for classification
    def quantile_binary_loss(y_pred, y_true, quantiles=[0.1, 0.5, 0.9]):
        """Loss function combining quantile loss with binary cross entropy."""
        # For a binary target (0 or 1), we use the middle quantile (p50) for BCE
        # And standard quantile loss for all three quantiles
        
        # Extract p50 prediction (middle quantile)
        p50 = y_pred[:, 1]
        
        # Sigmoid to get probability and BCE loss
        prob = torch.sigmoid(p50)
        bce_loss = F.binary_cross_entropy(prob, y_true.squeeze())
        
        # Quantile losses
        q_losses = []
        for i, q in enumerate(quantiles):
            errors = y_true - y_pred[:, i]
            q_losses.append(torch.max(q * errors, (q - 1) * errors))
        
        quantile_loss = torch.mean(torch.sum(torch.stack(q_losses), dim=0))
        
        # Combine losses (emphasize BCE since it's our primary objective)
        return bce_loss * 10.0 + quantile_loss
    
    # Define optimizer with weight decay
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=CONFIG['LEARNING_RATE'],
        weight_decay=CONFIG['WEIGHT_DECAY']
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5,
        verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    # Create directory for models if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    start_time = time.time()
    
    for epoch in range(CONFIG['EPOCHS']):
        # Training phase
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Get batch data
            past_data = batch['past_data'].to(device)
            static = batch['static'].to(device)
            target = batch['target'].to(device).unsqueeze(1)
            
            # Forward pass
            outputs = model(past_data, static)
            loss = quantile_binary_loss(outputs['quantiles'], target)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['GRADIENT_CLIP'])
            
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        predictions = []
        actuals = []
        
        with torch.no_grad():
            for batch in test_loader:
                past_data = batch['past_data'].to(device)
                static = batch['static'].to(device)
                target = batch['target'].to(device).unsqueeze(1)
                
                outputs = model(past_data, static)
                loss = quantile_binary_loss(outputs['quantiles'], target)
                val_loss += loss.item()
                
                # Store predictions and actuals for accuracy calculation
                p50 = torch.sigmoid(outputs['p50']).cpu().numpy()
                predictions.extend((p50 > 0.5).astype(int))
                actuals.extend(target.cpu().numpy())
        
        # Calculate accuracy
        predictions = np.array(predictions).flatten()
        actuals = np.array(actuals).flatten()
        accuracy = np.mean(predictions == actuals)
        
        # Calculate average loss
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        
        # Store losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        elapsed = time.time() - start_time
        if verbose:
            print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']}, "
                 f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, "
                 f"Accuracy: {accuracy:.4f}, LR: {current_lr:.6f}, "
                 f"Time: {elapsed:.1f}s")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'models/tft_model_improved.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    model.load_state_dict(torch.load('models/tft_model_improved.pth'))
    
    # Plot training curve
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curve.png')
    
    if verbose:
        print(f"Model training complete. Best validation loss: {best_val_loss:.6f}")
    
    return model, test_dataset, test_dates[CONFIG['SEQUENCE_LENGTH'] + CONFIG['FORECAST_HORIZON'] - 1:]

def evaluate_model(model, test_dataset, test_dates, known_cols, observed_cols, verbose=True):
    """Evaluate the model and generate predictions."""
    model.eval()
    
    # Create dataloader without shuffling to maintain order
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Collect predictions and actual values
    all_p50 = []
    all_p10 = []
    all_p90 = []
    all_targets = []
    all_binary_preds = []
    all_attention_weights = []
    all_static_weights = []
    all_known_weights = []
    all_observed_weights = []
    
    with torch.no_grad():
        for batch in test_loader:
            past_data = batch['past_data'].to(device)
            static = batch['static'].to(device)
            targets = batch['target']
            
            # Get model outputs
            outputs = model(past_data, static)
            
            # Extract predictions and weights
            p50 = torch.sigmoid(outputs['p50']).cpu().numpy()  # Sigmoid for binary target
            p10 = torch.sigmoid(outputs['p10']).cpu().numpy()
            p90 = torch.sigmoid(outputs['p90']).cpu().numpy()
            
            # Binary prediction
            binary_preds = (p50 > 0.5).astype(int)
            
            # Store predictions and targets
            all_p50.extend(p50)
            all_p10.extend(p10)
            all_p90.extend(p90)
            all_targets.extend(targets.numpy())
            all_binary_preds.extend(binary_preds)
            
            # Store weights for interpretability
            if outputs['attention_weights'] is not None:
                all_attention_weights.append(outputs['attention_weights'].cpu().numpy())
            
            if outputs['static_weights'] is not None:
                all_static_weights.append(outputs['static_weights'])
            
            if outputs['known_weights'] is not None:
                all_known_weights.append(outputs['known_weights'])
            
            if outputs['observed_weights'] is not None:
                all_observed_weights.append(outputs['observed_weights'])
    
    # Convert to numpy arrays
    all_p50 = np.array(all_p50).flatten()
    all_p10 = np.array(all_p10).flatten()
    all_p90 = np.array(all_p90).flatten()
    all_targets = np.array(all_targets).flatten()
    all_binary_preds = np.array(all_binary_preds).flatten()
    
    # Calculate accuracy
    accuracy = np.mean(all_binary_preds == all_targets)
    
    if verbose:
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Generated {len(all_p50)} predictions")
    
    # Generate trading signals based on p50 predictions
    signals = np.zeros(len(all_p50))
    signals[all_p50 > 0.6] = 1    # Buy signal when probability > 0.6
    signals[all_p50 < 0.4] = -1   # Sell signal when probability < 0.4
    
    # Calculate confidence from prediction probabilities
    # Higher confidence when closer to 0 or 1
    confidence = np.maximum(all_p50, 1 - all_p50)
    
    # Ensure we have some signals
    buy_signals = np.sum(signals == 1)
    sell_signals = np.sum(signals == -1)
    
    if verbose:
        print(f"Generated {buy_signals} buy signals and {sell_signals} sell signals")
    
    if buy_signals + sell_signals < 5:
        if verbose:
            print("WARNING: Not enough trading signals. Adjusting thresholds...")
        # Adjust thresholds to get more signals
        signals[all_p50 > 0.55] = 1    # Buy signal with lower threshold
        signals[all_p50 < 0.45] = -1   # Sell signal with higher threshold
        
        buy_signals = np.sum(signals == 1)
        sell_signals = np.sum(signals == -1)
        if verbose:
            print(f"After adjustment: {buy_signals} buy signals and {sell_signals} sell signals")
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': all_targets,
        'Predicted_Prob': all_p50,
        'P10': all_p10,
        'P90': all_p90,
        'Binary_Prediction': all_binary_preds,
        'Signal': signals,
        'Confidence': confidence
    })
    
    # Add price data for visualization
    price_df = load_and_prepare_data(verbose=False)
    
    # Add closing prices to results
    results_df['Close'] = np.nan
    for i, date in enumerate(results_df['Date']):
        if date in price_df.index:
            results_df.loc[i, 'Close'] = price_df.loc[date, 'Close']
    
    # Add technical indicators to results
    for indicator in ['EMA_8', 'SMA_200']:
        if indicator in price_df.columns:
            results_df[indicator] = np.nan
            for i, date in enumerate(results_df['Date']):
                if date in price_df.index:
                    results_df.loc[i, indicator] = price_df.loc[date, indicator]
    
    # Add sentiment to results
    for sentiment in ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']:
        if sentiment in price_df.columns:
            results_df[sentiment] = np.nan
            for i, date in enumerate(results_df['Date']):
                if date in price_df.index:
                    results_df.loc[i, sentiment] = price_df.loc[date, sentiment]
    
    # Plot results
    plt.figure(figsize=(15, 10))
    
    # Plot 1: Closing Prices with Buy/Sell Signals
    plt.subplot(2, 1, 1)
    plt.plot(results_df['Date'], results_df['Close'], label='Close Price', color='blue')
    
    if 'EMA_8' in results_df.columns and not results_df['EMA_8'].isna().all():
        plt.plot(results_df['Date'], results_df['EMA_8'], 
                label='8-day EMA', color='purple', linestyle='--', alpha=0.7)
    
    if 'SMA_200' in results_df.columns and not results_df['SMA_200'].isna().all():
        plt.plot(results_df['Date'], results_df['SMA_200'], 
                label='200-day SMA', color='gray', linestyle='--', alpha=0.7)
    
    # Plot buy/sell signals
    buy_signals = results_df['Signal'] == 1
    sell_signals = results_df['Signal'] == -1
    
    if buy_signals.any():
        plt.scatter(results_df['Date'][buy_signals], results_df['Close'][buy_signals], 
                color='green', marker='^', s=100, label='Buy Signal')
    
    if sell_signals.any():
        plt.scatter(results_df['Date'][sell_signals], results_df['Close'][sell_signals], 
                color='red', marker='v', s=100, label='Sell Signal')
    
    plt.title('AAPL Stock Price with Trading Signals (TFT Model)')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Prediction Probabilities
    plt.subplot(2, 1, 2)
    plt.plot(results_df['Date'], results_df['Predicted_Prob'], label='Up Probability', color='orange')
    plt.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    plt.axhline(y=0.6, color='green', linestyle='--', alpha=0.5, label='Buy Threshold (0.6)')
    plt.axhline(y=0.4, color='red', linestyle='--', alpha=0.5, label='Sell Threshold (0.4)')
    
    # Shade the confidence interval
    plt.fill_between(results_df['Date'], results_df['P10'].values, 
                    results_df['P90'].values, color='orange', alpha=0.2, 
                    label='P10-P90 Interval')
    
    # Add actual outcomes
    up_days = results_df['Actual'] == 1
    down_days = results_df['Actual'] == 0
    
    if up_days.any():
        plt.scatter(results_df['Date'][up_days], [1.05] * sum(up_days), 
                   color='green', marker='|', s=50, label='Actual Up')
    
    if down_days.any():
        plt.scatter(results_df['Date'][down_days], [-0.05] * sum(down_days), 
                   color='red', marker='|', s=50, label='Actual Down')
    
    plt.title('Predicted Probability of Price Increase')
    plt.ylabel('Probability')
    plt.ylim(-0.1, 1.1)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tft_predictions_improved.png')
    
    # Calculate variable importance from weights
    # Make a deep copy to be safe
    feature_importance = CONFIG['feature_importance'].copy()
    
    # Analyze weights if available
    if all_known_weights:
        known_weights_avg = np.mean(all_known_weights, axis=0)
        for i, col in enumerate(known_cols):
            if i < len(known_weights_avg):
                feature_importance[f'known_{col}'] = known_weights_avg[i]
    
    if all_observed_weights:
        observed_weights_avg = np.mean(all_observed_weights, axis=0)
        for i, col in enumerate(observed_cols):
            if i < len(observed_weights_avg):
                feature_importance[f'observed_{col}'] = observed_weights_avg[i]
    
    # Plot feature importance if available
    if isinstance(feature_importance, dict) and 'known' in feature_importance and 'observed' in feature_importance:
        plt.figure(figsize=(12, 6))
        # Convert lists to dict with values
        all_features = {}
        
        # Add known features
        for i, feat in enumerate(feature_importance['known']):
            try:
                if all_known_weights:
                    all_features[f"known_{feat}"] = np.mean(all_known_weights, axis=0)[i]
                else:
                    all_features[f"known_{feat}"] = 0
            except:
                all_features[f"known_{feat}"] = 0
        
        # Add observed features
        for i, feat in enumerate(feature_importance['observed']):
            try:
                if all_observed_weights:
                    all_features[f"observed_{feat}"] = np.mean(all_observed_weights, axis=0)[i]
                else:
                    all_features[f"observed_{feat}"] = 0
            except:
                all_features[f"observed_{feat}"] = 0
        
        # Sort features by importance
        sorted_features = dict(sorted(all_features.items(), key=lambda item: item[1], reverse=True))
        
        # Plot
        plt.figure(figsize=(12, 8))
        feature_names = list(sorted_features.keys())
        feature_values = list(sorted_features.values())
        
        # Use different colors for known and observed features
        colors = ['skyblue' if 'known_' in name else 'salmon' for name in feature_names]
        
        plt.barh(feature_names, feature_values, color=colors)
        plt.xlabel('Importance')
        plt.title('Feature Importance')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('feature_importance.png')
    
    # Save predictions to CSV
    results_df.to_csv('tft_predictions_improved.csv', index=False)
    if verbose:
        print("Predictions saved to tft_predictions_improved.csv")
    
    return results_df

def main():
    """Main function to run the TFT model."""
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Record start time
    start_time = time.time()
    
    # Load data
    df = load_and_prepare_data()
    
    # Prepare data for TFT
    data, static_features, dates, num_static, num_known, num_observed = prepare_data_for_tft(df)
    
    print(f"Data prepared with {num_static} static features, {num_known} known features, and {num_observed} observed features")
    print(f"Known features: {CONFIG.get('known_cols', [])}")
    print(f"Observed features: {CONFIG.get('observed_cols', [])}")
    
    # Train the model
    model, test_dataset, test_dates = train_model(
        data, static_features, dates, num_static, num_known, num_observed
    )
    
    # Evaluate model and generate predictions
    results_df = evaluate_model(
        model, test_dataset, test_dates, 
        CONFIG.get('known_cols', []),
        CONFIG.get('observed_cols', [])
    )
    
    # Print total execution time
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds ({execution_time/60:.2f} minutes)")
    
    print("\nTrading simulation can be executed using tft_trading_simulation.py")
    print("IMPORTANT: Update the predictions file to 'tft_predictions_improved.csv' in the simulation script")

if __name__ == "__main__":
    main()
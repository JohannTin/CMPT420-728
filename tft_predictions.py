import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset, DataLoader
import datetime
import os

# Configuration
CONFIG = {
    'SYMBOL': 'AAPL',
    'SEQUENCE_LENGTH': 30,
    'TRAIN_SIZE_RATIO': 0.85,
    'EPOCHS': 30,
    'BATCH_SIZE': 64,
    'LEARNING_RATE': 0.001,
    'HIDDEN_SIZE': 64,
    'NUM_HEADS': 4,
    'NUM_QUANTILES': 3,  # for P10, P50, P90
    'DROPOUT_RATE': 0.2,
    'CONFIDENCE_THRESHOLD': 0.7,
    'START_DATE': '2010-01-01',
    'END_DATE': '2025-01-01'
}

# Ensure deterministic behavior
torch.manual_seed(42)
np.random.seed(42)

class TimeSeriesDataset(Dataset):
    """Dataset for time series forecasting with TFT."""
    
    def __init__(self, data, static_covariates, seq_length, target_idx=0):
        self.data = data
        self.static_covariates = static_covariates
        self.seq_length = seq_length
        self.target_idx = target_idx
        
    def __len__(self):
        return len(self.data) - self.seq_length
    
    def __getitem__(self, idx):
        # Get sequence of data
        past_data = self.data[idx:idx + self.seq_length]
        
        # Target is the next step's target variable
        target = self.data[idx + self.seq_length, self.target_idx]
        
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
        flat_inputs = []
        for inp in inputs:
            flat_inputs.append(inp)
        
        # Check if context needs to be added
        if context is not None:
            all_inputs = flat_inputs + [context]
        else:
            all_inputs = flat_inputs
        
        # Concatenate along feature dimension
        weight_input = torch.cat(all_inputs, dim=-1)
        
        # Generate weights using the weight GRN
        weights = self.weight_grn(weight_input)
        weights = F.softmax(weights, dim=-1)
        
        # Weight the variable outputs
        combined_output = torch.zeros_like(var_outputs[0])
        for i, output in enumerate(var_outputs):
            # Extract the weight for this variable
            var_weight = weights[..., i].unsqueeze(-1)
            # Multiply the variable output by its weight
            combined_output = combined_output + var_weight * output
        
        return combined_output, weights

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
        
        # Project values (shared across all heads)
        projected_values = self.value_projection(values)
        
        # Initialize outputs
        outputs = torch.zeros(batch_size, queries.shape[1], self.head_size, device=queries.device)
        attention_weights = torch.zeros(batch_size, self.num_heads, queries.shape[1], keys.shape[1], device=queries.device)
        
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
                 hidden_size=64,
                 num_heads=4,
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
        )
        
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
        
        self.decoder_lstm = nn.LSTM(
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
        
        # Process each time step
        for t in range(seq_len):
            # Get features for this time step
            features = x[:, t, :]
            
            # Split into known and observed features
            known_inputs = [features[:, i:i+1] for i in range(self.num_known_features)]
            observed_inputs = [features[:, self.num_known_features+i:self.num_known_features+i+1] 
                              for i in range(self.num_observed_features)]
            
            # Process known inputs
            if known_inputs:
                known_embedding, known_weights = self.known_vsn(known_inputs, context_var_selection)
            else:
                known_embedding = torch.zeros(batch_size, self.hidden_size, device=x.device)
                known_weights = None
            
            # Process observed inputs
            observed_embedding, observed_weights = self.observed_vsn(observed_inputs, context_var_selection)
            
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
            'p50': quantiles[:, 1:2],  # Middle quantile is usually p50
            'attention_weights': attn_weights,
            'static_weights': static_weights
        }

def load_and_prepare_data():
    """Load and prepare both price and sentiment data."""
    # Load price data with indicators
    print("Loading data from appl_data_with_indicators.csv...")
    df_price = pd.read_csv('appl_data_with_indicators.csv')
    df_price['Date'] = pd.to_datetime(df_price['Unnamed: 0'])
    df_price.set_index('Date', inplace=True)
    df_price = df_price.sort_index()
    
    # Load sentiment data
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
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    
    return df

def prepare_data_for_tft(df):
    """Prepare data for TFT model."""
    # Debug information
    print("DataFrame shape:", df.shape)
    print("DataFrame columns:", df.columns.tolist())
    
    # Define feature groups
    target_col = 'Close'
    
    # Check if we have duplicate columns (common issue)
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"Warning: Duplicate columns found: {duplicate_cols}")
        # Create a new DataFrame with deduplicated columns
        df = df.loc[:, ~df.columns.duplicated()]
        print("After deduplication, shape:", df.shape)
    
    # Known features (calendar features)
    known_cols = ['day_of_week', 'day_of_month', 'month']
    
    # Observed features (past measurements we can observe)
    observed_cols = ['Volume', 'sentiment_positive', 'sentiment_negative']
    
    # If Close is in the columns, add it to observed features
    if target_col in df.columns:
        if target_col not in observed_cols:
            observed_cols.append(target_col)
    else:
        print(f"Warning: Target column '{target_col}' not found!")
        return None, None, None, None, None, None
    
    # Verify columns exist
    all_cols = known_cols + observed_cols
    for col in all_cols[:]:  # Use slice to allow modification during iteration
        if col not in df.columns:
            print(f"Warning: Column {col} not found in data. Removing from features.")
            if col in known_cols:
                known_cols.remove(col)
            if col in observed_cols:
                observed_cols.remove(col)
    
    # Clean data - convert to numeric and drop NaN
    feature_cols = known_cols + observed_cols
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df_clean = df[feature_cols].copy()
    df_clean = df_clean.dropna()
    
    # Create a target column that's the next day's closing price
    next_day_close = df_clean[target_col].shift(-1)
    df_clean['target'] = next_day_close
    
    # Drop rows with NaN target
    df_clean = df_clean.dropna()
    
    # Normalize data
    scaler = StandardScaler()
    data = scaler.fit_transform(df_clean[feature_cols + ['target']])
    
    # Create static features (simple for now - just 1 for AAPL)
    static_features = [1]  # Stock ID
    
    # Define constants for model
    num_known = len(known_cols)
    num_observed = len(observed_cols)
    num_static = len(static_features)
    
    # Save info for later
    CONFIG['target_scaler'] = scaler
    CONFIG['target_idx'] = len(feature_cols)  # Target is the last column
    
    return data, static_features, df_clean.index, num_static, num_known, num_observed

def train_model(data, static_features, dates, num_static, num_known, num_observed):
    """Train TFT model."""
    # Split data into train and test
    train_size = int(len(data) * CONFIG['TRAIN_SIZE_RATIO'])
    train_data = data[:train_size]
    test_data = data[train_size:]
    train_dates = dates[:train_size]
    test_dates = dates[train_size:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(
        data=train_data,
        static_covariates=static_features,
        seq_length=CONFIG['SEQUENCE_LENGTH'],
        target_idx=CONFIG['target_idx']
    )
    
    test_dataset = TimeSeriesDataset(
        data=test_data,
        static_covariates=static_features,
        seq_length=CONFIG['SEQUENCE_LENGTH'],
        target_idx=CONFIG['target_idx']
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
    )
    
    # Define loss function for quantile regression
    def quantile_loss(y_pred, y_true, quantiles=[0.1, 0.5, 0.9]):
        losses = []
        for i, q in enumerate(quantiles):
            errors = y_true - y_pred[:, i]
            losses.append(torch.max(q * errors, (q - 1) * errors))
        return torch.mean(torch.sum(torch.stack(losses), dim=0))
    
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['LEARNING_RATE'])
    
    # Training loop
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(CONFIG['EPOCHS']):
        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Get batch data
            past_data = batch['past_data']
            static = batch['static']
            target = batch['target'].unsqueeze(1)
            
            # Forward pass
            outputs = model(past_data, static)
            loss = quantile_loss(outputs['quantiles'], target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                past_data = batch['past_data']
                static = batch['static']
                target = batch['target'].unsqueeze(1)
                
                outputs = model(past_data, static)
                loss = quantile_loss(outputs['quantiles'], target)
                val_loss += loss.item()
        
        # Calculate average loss
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        
        # Print progress
        print(f"Epoch {epoch+1}/{CONFIG['EPOCHS']}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), 'models/tft_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model
    model.load_state_dict(torch.load('models/tft_model.pth'))
    
    return model, test_dataset, test_dates[CONFIG['SEQUENCE_LENGTH']:]

def calculate_confidence(predictions, actuals):
    """Calculate confidence based on prediction error."""
    errors = np.abs(predictions - actuals)
    max_error = np.max(actuals) - np.min(actuals)
    confidence = 1 - (errors / max_error)
    return confidence

def generate_trading_signals(predictions, actuals, model_uncertainty, confidence_threshold):
    """Generate trading signals based on predictions and confidence."""
    # Calculate confidence from prediction errors
    confidence = calculate_confidence(predictions, actuals)
    
    # Adjust confidence based on model uncertainty
    adjusted_confidence = confidence * (1 - model_uncertainty)
    
    # Initialize signals
    signals = np.zeros(len(predictions))
    
    # Generate buy/sell signals
    high_confidence = adjusted_confidence >= confidence_threshold
    price_diff = predictions - actuals
    
    # Buy when predicted price is higher than current
    signals[high_confidence & (price_diff > 0)] = 1
    
    # Sell when predicted price is lower than current
    signals[high_confidence & (price_diff < 0)] = -1
    
    return signals, adjusted_confidence

def evaluate_model(model, test_dataset, test_dates):
    """Evaluate the model and generate predictions."""
    model.eval()
    
    # Create dataloader without shuffling to maintain order
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Collect predictions and actual values
    all_predictions = []
    all_targets = []
    all_attention_weights = []
    all_model_uncertainty = []
    
    with torch.no_grad():
        for batch in test_loader:
            past_data = batch['past_data']
            static = batch['static']
            targets = batch['target']
            
            # Get model outputs
            outputs = model(past_data, static)
            
            # Extract p50 predictions, attention weights, and uncertainty (p90-p10)
            p50_predictions = outputs['p50'].squeeze(-1).numpy()
            attention_weights = outputs['attention_weights'].numpy()
            
            # Calculate uncertainty as the difference between p90 and p10 predictions
            p10 = outputs['quantiles'][:, 0].numpy()
            p90 = outputs['quantiles'][:, 2].numpy()
            uncertainty = (p90 - p10) / 2  # Half the prediction interval width
            
            # Normalize uncertainty to [0, 1]
            max_uncertainty = np.max(uncertainty) if np.max(uncertainty) > 0 else 1
            normalized_uncertainty = uncertainty / max_uncertainty
            
            # Collect results
            all_predictions.extend(p50_predictions)
            all_targets.extend(targets.numpy())
            all_attention_weights.append(attention_weights)
            all_model_uncertainty.extend(normalized_uncertainty)
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    actuals = np.array(all_targets)
    model_uncertainty = np.array(all_model_uncertainty)
    
    # Inverse transform predictions and actuals using target column from scaler
    target_scaler = CONFIG['target_scaler']
    
    # Reshape for inverse transform
    temp_array = np.zeros((len(predictions), target_scaler.n_features_in_))
    temp_array[:, CONFIG['target_idx']] = predictions
    
    target_temp = np.zeros((len(actuals), target_scaler.n_features_in_))
    target_temp[:, CONFIG['target_idx']] = actuals
    
    # Inverse transform
    temp_array_inv = target_scaler.inverse_transform(temp_array)
    target_temp_inv = target_scaler.inverse_transform(target_temp)
    
    # Extract the target column
    predictions_inv = temp_array_inv[:, CONFIG['target_idx']]
    actuals_inv = target_temp_inv[:, CONFIG['target_idx']]
    
    # Generate trading signals
    signals, confidence = generate_trading_signals(
        predictions=predictions_inv,
        actuals=actuals_inv,
        model_uncertainty=model_uncertainty,
        confidence_threshold=CONFIG['CONFIDENCE_THRESHOLD']
    )
    
    # Calculate RMSE
    rmse = np.sqrt(np.mean((predictions_inv - actuals_inv) ** 2))
    print(f"Test RMSE: {rmse:.2f}")
    
    # Create DataFrame with results
    results_df = pd.DataFrame({
        'Date': test_dates,
        'Actual': actuals_inv,
        'Predicted': predictions_inv,
        'Signal': signals,
        'Confidence': confidence
    })
    
    # Load additional data for visualization
    temp_df = load_and_prepare_data()
    
    # Add technical indicators and sentiment
    results_df['EMA_8'] = np.nan
    results_df['SMA_200'] = np.nan
    results_df['Sentiment_Positive'] = np.nan
    results_df['Sentiment_Neutral'] = np.nan
    results_df['Sentiment_Negative'] = np.nan
    
    for i, date in enumerate(results_df['Date']):
        if date in temp_df.index:
            if 'EMA_8' in temp_df.columns:
                results_df.loc[i, 'EMA_8'] = temp_df.loc[date, 'EMA_8']
            if 'SMA_200' in temp_df.columns:
                results_df.loc[i, 'SMA_200'] = temp_df.loc[date, 'SMA_200']
            if 'sentiment_positive' in temp_df.columns:
                results_df.loc[i, 'Sentiment_Positive'] = temp_df.loc[date, 'sentiment_positive']
            if 'sentiment_neutral' in temp_df.columns:
                results_df.loc[i, 'Sentiment_Neutral'] = temp_df.loc[date, 'sentiment_neutral']
            if 'sentiment_negative' in temp_df.columns:
                results_df.loc[i, 'Sentiment_Negative'] = temp_df.loc[date, 'sentiment_negative']
    
    # Plot results
    plt.figure(figsize=(15, 7))
    plt.plot(results_df['Date'], results_df['Actual'], label='Actual', color='blue')
    plt.plot(results_df['Date'], results_df['Predicted'], label='Predicted', color='orange')
    
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
        plt.scatter(results_df['Date'][buy_signals], results_df['Actual'][buy_signals], 
                color='green', marker='^', s=100, label='Buy Signal')
    
    if sell_signals.any():
        plt.scatter(results_df['Date'][sell_signals], results_df['Actual'][sell_signals], 
                color='red', marker='v', s=100, label='Sell Signal')
    
    plt.title('AAPL Stock Price Prediction with Trading Signals (TFT Model)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('tft_predictions.png')
    
    # Save predictions to CSV
    results_df.to_csv('tft_predictions.csv', index=False)
    print("Predictions saved to tft_predictions.csv")
    
    return results_df

def main():
    # Create model directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Load data
    df = load_and_prepare_data()
    
    # Prepare data for TFT
    data, static_features, dates, num_static, num_known, num_observed = prepare_data_for_tft(df)
    
    print(f"Data prepared with {num_static} static features, {num_known} known features, and {num_observed} observed features")
    
    # Train the model
    model, test_dataset, test_dates = train_model(data, static_features, dates, num_static, num_known, num_observed)
    
    # Evaluate model and generate predictions
    results_df = evaluate_model(model, test_dataset, test_dates)
    
    print("\nTrading simulation will be executed using tft_trading_simulation.py")

if __name__ == "__main__":
    main()
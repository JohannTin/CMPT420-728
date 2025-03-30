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
import itertools
from tqdm import tqdm
import time
import json

# Configuration
CONFIG = {
    "SYMBOL": "AAPL",
    "SEQUENCE_LENGTH": 30,
    "TRAIN_SIZE_RATIO": 0.9,
    "EPOCHS": 50,
    "BATCH_SIZE": 64,
    "LEARNING_RATE": 0.001,
    "HIDDEN_SIZE": 128,
    "NUM_HEADS": 8,
    "NUM_QUANTILES": 3,  # for P10, P50, P90
    "DROPOUT_RATE": 0.2,
    "CONFIDENCE_THRESHOLD": 0.7,
    "START_DATE": "2010-01-01",
    "END_DATE": "2025-03-21",
}

# Grid search parameters - expanded to explore learning rate and model complexity
PARAM_GRID = {
    "SEQUENCE_LENGTH": [30],
    "BATCH_SIZE": [64],
    "LEARNING_RATE": [0.0001, 0.001, 0.01],  # Added more learning rate options
    "HIDDEN_SIZE": [64, 128, 256],  # Added more hidden size options
    "NUM_HEADS": [4, 8],
    "DROPOUT_RATE": [0.1, 0.2],  # Reduced dropout options
    "NUM_QUANTILES": [3],
    "CONFIDENCE_THRESHOLD": [0.7],
    "EPOCHS": [100],  # Increased max epochs
    "TRAIN_SIZE_RATIO": [0.9],
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
        past_data = self.data[idx : idx + self.seq_length]

        # Target is the next step's target variable
        target = self.data[idx + self.seq_length, self.target_idx]

        # Static covariates are the same for each sequence
        static = self.static_covariates.copy()

        return {
            "past_data": torch.tensor(past_data, dtype=torch.float32),
            "static": torch.tensor(static, dtype=torch.float32),
            "target": torch.tensor(target, dtype=torch.float32),
        }


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network as described in the TFT paper (Section 4.1)."""

    def __init__(
        self, input_size, hidden_size, output_size, dropout_rate=0.1, context_size=None
    ):
        super(GatedResidualNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.context_size = context_size

        # Dense layers for main processing
        self.W2 = nn.Linear(input_size, hidden_size)  # W_2,ω in Eq. 4
        self.W1 = nn.Linear(hidden_size, output_size)  # W_1,ω in Eq. 3

        # Context projection if context is provided
        if context_size is not None:
            self.W3 = nn.Linear(context_size, hidden_size)  # W_3,ω in Eq. 4

        # GLU layers (Eq. 5)
        self.W4 = nn.Linear(output_size, output_size)  # W_4,ω for gate
        self.W5 = nn.Linear(output_size, output_size)  # W_5,ω for values

        # Residual connection (skip connection)
        if input_size != output_size:
            self.skip_layer = nn.Linear(input_size, output_size)
        else:
            self.skip_layer = None

        # Layer normalization
        self.layer_norm = nn.LayerNorm(output_size)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, a, c=None):
        """
        Forward pass through the GRN.
        Args:
            a: Primary input
            c: Optional context vector
        """
        # Compute η2 (Eq. 4)
        eta2 = self.W2(a)
        if self.context_size is not None and c is not None:
            eta2 = eta2 + self.W3(c)
        eta2 = F.elu(eta2)  # ELU activation

        # Compute η1 (Eq. 3)
        eta1 = self.W1(eta2)

        # Apply dropout to η1
        eta1 = self.dropout(eta1)

        # GLU mechanism (Eq. 5)
        gate = torch.sigmoid(self.W4(eta1))
        value = self.W5(eta1)
        glu_output = gate * value  # Element-wise multiplication

        # Skip connection
        if self.skip_layer is not None:
            skip = self.skip_layer(a)
        else:
            skip = a

        # Apply layer normalization and return (Eq. 2)
        output = self.layer_norm(skip + glu_output)

        return output


class VariableSelectionNetwork(nn.Module):
    """Variable selection network from TFT (Section 4.2)."""

    def __init__(
        self, input_sizes, hidden_size, output_size, dropout_rate=0.1, context_size=None
    ):
        super(VariableSelectionNetwork, self).__init__()
        self.hidden_size = hidden_size
        self.input_sizes = input_sizes
        self.num_inputs = len(input_sizes)
        self.context_size = context_size

        # GRN for variable weights
        weight_grn_input_size = sum(input_sizes)

        # If context is provided, include it in the input size
        if context_size is not None:
            self.weight_grn = GatedResidualNetwork(
                input_size=weight_grn_input_size,
                hidden_size=hidden_size,
                output_size=self.num_inputs,  # Output size is the number of variables (for weights)
                dropout_rate=dropout_rate,
                context_size=context_size,
            )
        else:
            self.weight_grn = GatedResidualNetwork(
                input_size=weight_grn_input_size,
                hidden_size=hidden_size,
                output_size=self.num_inputs,
                dropout_rate=dropout_rate,
            )

        # GRN for each variable
        self.variable_grns = nn.ModuleList(
            [
                GatedResidualNetwork(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    output_size=output_size,
                    dropout_rate=dropout_rate,
                )
                for input_size in input_sizes
            ]
        )

    def forward(self, inputs, context=None):
        """
        Forward pass through variable selection network.

        Args:
            inputs: List of input tensors, one for each variable
            context: Optional context tensor from static variables

        Returns:
            combined_output: Selected and processed variables
            weights: Variable selection weights for interpretability
        """
        # Verify we have the right number of inputs
        assert (
            len(inputs) == self.num_inputs
        ), f"Expected {self.num_inputs} inputs but got {len(inputs)}"

        # Process each variable with its own GRN (Eq. 7)
        var_outputs = [
            grn(var_input) for grn, var_input in zip(self.variable_grns, inputs)
        ]

        # Concatenate all inputs for weight computation
        flat_inputs = torch.cat(inputs, dim=-1)

        # Generate variable selection weights
        if self.context_size is not None and context is not None:
            # Get selection weights with context (Eq. 6)
            weights = self.weight_grn(flat_inputs, context)
        else:
            # Get selection weights without context
            weights = self.weight_grn(flat_inputs)

        # Apply softmax to get normalized variable weights (Eq. 6)
        weights = F.softmax(weights, dim=-1)

        # Weight and combine variable outputs (Eq. 8)
        combined_output = torch.zeros_like(var_outputs[0])
        for i, output in enumerate(var_outputs):
            # Extract weight for this variable
            var_weight = weights[..., i].unsqueeze(-1)
            # Apply the weight to the processed variable output
            combined_output = combined_output + var_weight * output

        return combined_output, weights


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable Multi-head Attention from TFT paper (Section 4.4)."""

    def __init__(self, hidden_size, num_heads, dropout_rate=0.1):
        super(InterpretableMultiHeadAttention, self).__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads

        # Create separate key and query projection for each head
        self.key_projections = nn.ModuleList(
            [nn.Linear(hidden_size, self.head_size) for _ in range(num_heads)]
        )
        self.query_projections = nn.ModuleList(
            [nn.Linear(hidden_size, self.head_size) for _ in range(num_heads)]
        )

        # Shared value projection across all heads (key difference from standard multi-head attention)
        self.value_projection = nn.Linear(hidden_size, self.head_size)

        # Output projection
        self.output_projection = nn.Linear(self.head_size, hidden_size)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, queries, keys, values, mask=None):
        """
        Forward pass with interpretable attention weights.

        Args:
            queries: Query tensor [batch_size, seq_len, hidden_size]
            keys: Key tensor [batch_size, seq_len, hidden_size]
            values: Value tensor [batch_size, seq_len, hidden_size]
            mask: Optional mask tensor [seq_len, seq_len]

        Returns:
            output: Attention output [batch_size, seq_len, hidden_size]
            attention_weights: Attention weights for interpretability [batch_size, seq_len, seq_len]
        """
        batch_size = queries.shape[0]
        seq_len = queries.shape[1]

        # Project values once (shared across all heads) - Eq. 14
        projected_values = self.value_projection(values)

        # Initialize attention weights storage for all heads
        head_outputs = []
        attention_weights_by_head = torch.zeros(
            batch_size, self.num_heads, seq_len, keys.shape[1], device=queries.device
        )

        # Process each head separately
        for h in range(self.num_heads):
            # Project queries and keys for this head
            projected_queries = self.query_projections[h](queries)
            projected_keys = self.key_projections[h](keys)

            # Calculate attention scores (scaled dot-product attention)
            # Eq. 10: A(Q,K) = Softmax(QK^T/sqrt(d_attn))
            scores = torch.matmul(projected_queries, projected_keys.transpose(-2, -1))
            scores = scores / (self.head_size**0.5)  # Scaling

            # Apply mask if provided (crucial for decoder masking)
            if mask is not None:
                scores = scores.masked_fill(mask == 0, -1e9)

            # Apply softmax to get attention weights
            attention = F.softmax(scores, dim=-1)
            attention = self.dropout(attention)

            # Save attention weights for interpretability
            attention_weights_by_head[:, h] = attention

            # Apply attention to values and collect outputs from each head
            head_output = torch.matmul(attention, projected_values)
            head_outputs.append(head_output)

        # Average over all heads (Eq. 15) - key difference from standard multi-head attention
        # which concatenates instead of averaging
        outputs = torch.stack(head_outputs).mean(dim=0)

        # Final projection
        outputs = self.output_projection(outputs)

        # Return average attention weights across heads for interpretability (Eq. 15)
        attention_weights = attention_weights_by_head.mean(dim=1)

        return outputs, attention_weights


class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer for time series forecasting."""

    def __init__(
        self,
        num_static_features=1,
        num_known_features=3,
        num_observed_features=24,  # Increased to handle all technical indicators
        hidden_size=128,
        num_heads=4,
        dropout_rate=0.1,
        num_quantiles=3,
    ):
        super(TemporalFusionTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.num_quantiles = num_quantiles

        # Feature counts
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
            dropout_rate=dropout_rate,
        )

        self.known_vsn = VariableSelectionNetwork(
            input_sizes=known_input_sizes,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate,
            context_size=hidden_size,  # Context from static variables
        )

        self.observed_vsn = VariableSelectionNetwork(
            input_sizes=observed_input_sizes,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate,
            context_size=hidden_size,  # Context from static variables
        )

        # Static context vectors for various purposes
        self.static_context_variable_selection = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate,
        )

        self.static_context_initial_hidden = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate,
        )

        self.static_context_initial_cell = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate,
        )

        self.static_context_enrichment = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate,
        )

        # Sequence-to-sequence layer (LSTM)
        self.encoder_lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )

        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, batch_first=True
        )

        # Skip connection gate for LSTM outputs (Eq. 17)
        self.post_lstm_gate_norm = nn.LayerNorm(hidden_size)
        self.post_lstm_gate_linear = nn.Linear(hidden_size, hidden_size)

        # Static enrichment layer (Eq. 18)
        self.static_enrichment = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate,
            context_size=hidden_size,
        )

        # Temporal attention layer
        self.multihead_attention = InterpretableMultiHeadAttention(
            hidden_size=hidden_size, num_heads=num_heads, dropout_rate=dropout_rate
        )

        # Skip connection gate for attention outputs (Eq. 20)
        self.post_attn_gate_norm = nn.LayerNorm(hidden_size)
        self.post_attn_gate_linear = nn.Linear(hidden_size, hidden_size)

        # Position-wise feed-forward layer (Eq. 21)
        self.position_wise_ff = GatedResidualNetwork(
            input_size=hidden_size,
            hidden_size=hidden_size,
            output_size=hidden_size,
            dropout_rate=dropout_rate,
        )

        # Skip connection over the entire transformer block (Eq. 22)
        self.post_tft_gate_norm = nn.LayerNorm(hidden_size)
        self.post_tft_gate_linear = nn.Linear(hidden_size, hidden_size)

        # Quantile prediction layers
        self.quantile_proj = nn.Linear(hidden_size, num_quantiles)

    def forward(self, x, static=None, mask=None):
        """
        Forward pass of TFT.

        Args:
            x: Input tensor of shape [batch_size, seq_len, input_size]
            static: Static features of shape [batch_size, num_static_features]
            mask: Optional mask for self-attention to ensure causal information flow

        Returns:
            Dictionary containing model outputs
        """
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # Process static features
        if static is not None:
            # Split static features
            static_inputs = [static[:, i : i + 1] for i in range(static.size(1))]

            # Apply variable selection to static features
            static_embedding, static_weights = self.static_vsn(static_inputs)

            # Generate context vectors for different parts of the network
            context_var_selection = self.static_context_variable_selection(
                static_embedding
            )
            context_init_hidden = self.static_context_initial_hidden(static_embedding)
            context_init_cell = self.static_context_initial_cell(static_embedding)
            context_enrichment = self.static_context_enrichment(static_embedding)
        else:
            # Default context vectors when no static features are provided
            static_embedding = torch.zeros(
                batch_size, self.hidden_size, device=x.device
            )
            context_var_selection = torch.zeros(
                batch_size, self.hidden_size, device=x.device
            )
            context_init_hidden = torch.zeros(
                batch_size, self.hidden_size, device=x.device
            )
            context_init_cell = torch.zeros(
                batch_size, self.hidden_size, device=x.device
            )
            context_enrichment = torch.zeros(
                batch_size, self.hidden_size, device=x.device
            )
            static_weights = None

        # Container for processed temporal features
        temporal_features = []
        known_weights_seq = []
        observed_weights_seq = []

        # Process each time step
        for t in range(seq_len):
            # Get features for this time step
            features = x[:, t, :]

            # Split into known and observed features
            known_inputs = [
                features[:, i : i + 1] for i in range(self.num_known_features)
            ]
            observed_inputs = [
                features[
                    :, self.num_known_features + i : self.num_known_features + i + 1
                ]
                for i in range(self.num_observed_features)
            ]

            # Process known inputs with variable selection
            if known_inputs and self.num_known_features > 0:
                known_embedding, known_weights = self.known_vsn(
                    known_inputs, context_var_selection
                )
                known_weights_seq.append(known_weights)
            else:
                known_embedding = torch.zeros(
                    batch_size, self.hidden_size, device=x.device
                )
                known_weights_seq.append(None)

            # Process observed inputs with variable selection
            if self.num_observed_features > 0:
                observed_embedding, observed_weights = self.observed_vsn(
                    observed_inputs, context_var_selection
                )
                observed_weights_seq.append(observed_weights)
            else:
                observed_embedding = torch.zeros(
                    batch_size, self.hidden_size, device=x.device
                )
                observed_weights_seq.append(None)

            # Combine embeddings
            combined_embedding = known_embedding + observed_embedding
            temporal_features.append(combined_embedding)

        # Stack temporal features
        temporal_features = torch.stack(
            temporal_features, dim=1
        )  # [batch_size, seq_len, hidden_size]

        # Initialize LSTM states with static context
        h0 = context_init_hidden.unsqueeze(0)  # Add sequence length dimension
        c0 = context_init_cell.unsqueeze(0)

        # Pass through encoder LSTM (sequence-to-sequence layer)
        lstm_output, (hidden, cell) = self.encoder_lstm(temporal_features, (h0, c0))

        # Gated skip connection for LSTM outputs (Eq. 17)
        lstm_gate = torch.sigmoid(self.post_lstm_gate_linear(lstm_output))
        lstm_output = self.post_lstm_gate_norm(
            temporal_features + lstm_gate * lstm_output
        )

        # Static enrichment for each time step (Eq. 18)
        enriched_output = torch.zeros_like(lstm_output)
        for t in range(seq_len):
            enriched_output[:, t] = self.static_enrichment(
                lstm_output[:, t], context_enrichment
            )

        # Create proper causal mask for self-attention if not provided
        # This ensures decoder masking as mentioned in the paper (Sec. 4.5.3)
        if mask is None:
            # Create causal mask for self-attention (lower triangular)
            mask = torch.tril(torch.ones(seq_len, seq_len)).to(x.device)

        # Apply self-attention (Eq. 19) with proper masking
        attn_output, attn_weights = self.multihead_attention(
            queries=enriched_output,
            keys=enriched_output,
            values=enriched_output,
            mask=mask,
        )

        # Gated skip connection for attention outputs (Eq. 20)
        attn_gate = torch.sigmoid(self.post_attn_gate_linear(attn_output))
        attn_output = self.post_attn_gate_norm(
            enriched_output + attn_gate * attn_output
        )

        # Position-wise feed-forward network (Eq. 21) applied to each time step
        ff_output = torch.zeros_like(attn_output)
        for t in range(seq_len):
            ff_output[:, t] = self.position_wise_ff(attn_output[:, t])

        # Gated skip connection over the entire transformer block (Eq. 22)
        transformer_gate = torch.sigmoid(self.post_tft_gate_linear(ff_output))
        transformer_out = self.post_tft_gate_norm(
            lstm_output + transformer_gate * ff_output
        )

        # Generate quantile predictions for the last timestep
        final_hidden = transformer_out[:, -1]
        quantiles = self.quantile_proj(final_hidden)

        # Return model outputs
        return {
            "quantiles": quantiles,
            "p50": quantiles[:, 1:2],  # Middle quantile is usually p50
            "p10": quantiles[:, 0:1],  # Lower quantile is p10
            "p90": quantiles[:, 2:3],  # Upper quantile is p90
            "attention_weights": attn_weights,
            "static_weights": static_weights,
            "known_weights": known_weights_seq,
            "observed_weights": observed_weights_seq,
        }


def load_and_prepare_data():
    """Load and prepare both price and sentiment data."""
    # Load price data with indicators
    print("Loading data from appl_data_with_indicators.csv...")
    df_price = pd.read_csv("appl_data_with_indicators.csv")
    df_price["Date"] = pd.to_datetime(df_price["Unnamed: 0"])
    df_price.set_index("Date", inplace=True)
    df_price = df_price.sort_index()

    # Load sentiment data
    print("Loading data from sentiment_analysis_detailed.csv...")
    df_sentiment = pd.read_csv("sentiment_analysis_detailed.csv")
    # Convert published_at to datetime and extract date more robustly
    df_sentiment["Date"] = pd.to_datetime(
        df_sentiment["published_at"], utc=True
    ).dt.strftime("%Y-%m-%d")
    df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"])
    df_sentiment = (
        df_sentiment.groupby("Date")
        .agg(
            {
                "sentiment_positive": "mean",
                "sentiment_neutral": "mean",
                "sentiment_negative": "mean",
            }
        )
        .reset_index()
    )
    df_sentiment["Date"] = pd.to_datetime(df_sentiment["Date"])
    df_sentiment.set_index("Date", inplace=True)

    # Merge price and sentiment data
    df = df_price.join(df_sentiment, how="left")

    # Forward fill sentiment scores for days without news
    df[["sentiment_positive", "sentiment_neutral", "sentiment_negative"]] = df[
        ["sentiment_positive", "sentiment_neutral", "sentiment_negative"]
    ].ffill()

    # If there are still NaN values at the start, backward fill
    df[["sentiment_positive", "sentiment_neutral", "sentiment_negative"]] = df[
        ["sentiment_positive", "sentiment_neutral", "sentiment_negative"]
    ].bfill()

    # Fill any remaining NaN values with neutral sentiment
    df[["sentiment_positive", "sentiment_neutral", "sentiment_negative"]] = df[
        ["sentiment_positive", "sentiment_neutral", "sentiment_negative"]
    ].fillna(1 / 3)

    # Filter date range
    df = df[(df.index >= CONFIG["START_DATE"]) & (df.index <= CONFIG["END_DATE"])]

    # Add date features (known inputs)
    df["day_of_week"] = df.index.dayofweek
    df["day_of_month"] = df.index.day
    df["month"] = df.index.month

    return df


def prepare_data_for_tft(df):
    """Prepare data for TFT model with proper feature classification."""
    # Debug information
    print("DataFrame shape:", df.shape)
    print("DataFrame columns:", df.columns.tolist())

    # Define target column
    target_col = "Close"

    # Check for duplicate columns
    duplicate_cols = df.columns[df.columns.duplicated()].tolist()
    if duplicate_cols:
        print(f"Warning: Duplicate columns found: {duplicate_cols}")
        df = df.loc[:, ~df.columns.duplicated()]
        print("After deduplication, shape:", df.shape)

    # KNOWN FEATURES: Calendar features that can be predetermined
    known_cols = ["day_of_week", "day_of_month", "month"]

    # OBSERVED FEATURES: All technical indicators and price data
    observed_cols = [
        # Price data
        "Open",
        "High",
        "Low",
        "Close",
        "Volume",
        # Moving averages
        "SMA_50",
        "SMA_200",
        "EMA_8",
        "EMA_12",
        "EMA_26",
        # Momentum indicators
        "MACD",
        "Signal_Line",
        "MACD_Histogram",
        "RSI",
        # Stochastic oscillators
        "%K",
        "%D",
        # Range indicators
        "14-high",
        "14-low",
        # Volatility indicators
        "BB_middle",
        "BB_upper",
        "BB_lower",
        "TR",
        "ATR",
        # Volume indicators
        "OBV",
        # Sentiment indicators
        "sentiment_positive",
        "sentiment_negative",
        "sentiment_neutral",
    ]

    # Verify which columns actually exist in the DataFrame
    available_known_cols = [col for col in known_cols if col in df.columns]
    available_observed_cols = [col for col in observed_cols if col in df.columns]

    # Print missing columns
    missing_known = set(known_cols) - set(available_known_cols)
    missing_observed = set(observed_cols) - set(available_observed_cols)

    if missing_known:
        print(f"Warning: The following known features are missing: {missing_known}")
    if missing_observed:
        print(
            f"Warning: The following observed features are missing: {missing_observed}"
        )

    # Update with available columns
    known_cols = available_known_cols
    observed_cols = available_observed_cols

    # Ensure target column is included in observed features
    if target_col in df.columns and target_col not in observed_cols:
        observed_cols.append(target_col)

    # Handle columns with special characters
    renamed_columns = {}
    for col in list(observed_cols):
        if any(char in col for char in ["%", "-"]):
            # Create safe column name
            safe_name = col.replace("%", "pct").replace("-", "_")
            renamed_columns[col] = safe_name
            # Update observed_cols list
            observed_cols.remove(col)
            observed_cols.append(safe_name)

    # Rename columns if needed
    if renamed_columns:
        print(f"Renaming columns with special characters: {renamed_columns}")
        df = df.rename(columns=renamed_columns)

    # Clean data - convert to numeric and drop NaN
    feature_cols = known_cols + observed_cols
    for col in feature_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df_clean = df[feature_cols].copy()
    df_clean = df_clean.dropna()

    # Create a target column that's the next day's closing price
    next_day_close = df_clean[target_col].shift(-1)
    df_clean["target"] = next_day_close

    # Drop rows with NaN target
    df_clean = df_clean.dropna()

    # Normalize data using StandardScaler for better numerical stability
    scaler = StandardScaler()
    data = scaler.fit_transform(df_clean[feature_cols + ["target"]])

    # Create static features (expand beyond just a placeholder)
    # For example, you could add stock metadata, market cap category, sector, etc.
    # For now, we'll still use a placeholder but you should expand this
    static_features = [1]  # Replace with actual static features when available

    # Define counts for model initialization
    num_known = len(known_cols)
    num_observed = len(observed_cols)
    num_static = len(static_features)

    print(
        f"Final feature counts: {num_static} static, {num_known} known, {num_observed} observed"
    )

    # Save info for later
    CONFIG["target_scaler"] = scaler
    CONFIG["target_idx"] = len(feature_cols)  # Target is the last column
    CONFIG["feature_cols"] = feature_cols  # Save column names for interpretation

    return data, static_features, df_clean.index, num_static, num_known, num_observed


def quantile_loss(y_pred, y_true, quantiles=[0.1, 0.5, 0.9]):
    """
    Calculate the quantile loss for multiple quantiles.

    Args:
        y_pred: Predicted quantiles of shape [batch_size, num_quantiles]
        y_true: Actual target values of shape [batch_size, 1]
        quantiles: List of quantile values (default: [0.1, 0.5, 0.9])

    Returns:
        Total quantile loss summed across all quantiles
    """
    losses = []
    for i, q in enumerate(quantiles):
        errors = y_true - y_pred[:, i : i + 1]
        losses.append(torch.max(q * errors, (q - 1) * errors))

    # Return mean of losses summed across quantiles
    return torch.mean(torch.sum(torch.stack(losses), dim=0))


def train_model_with_params(
    data, static_features, dates, num_static, num_known, num_observed, config
):
    """
    Train TFT model with the specified parameters.
    Returns the best validation loss and trained model.
    """
    # Create train/test split
    train_size = int(len(data) * config["TRAIN_SIZE_RATIO"])
    train_data = data[:train_size]
    test_data = data[train_size:]
    train_dates = dates[:train_size]
    test_dates = dates[train_size:]

    # Create datasets with current sequence length
    train_dataset = TimeSeriesDataset(
        data=train_data,
        static_covariates=static_features,
        seq_length=config["SEQUENCE_LENGTH"],
        target_idx=config["target_idx"],
    )

    test_dataset = TimeSeriesDataset(
        data=test_data,
        static_covariates=static_features,
        seq_length=config["SEQUENCE_LENGTH"],
        target_idx=config["target_idx"],
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config["BATCH_SIZE"], shuffle=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config["BATCH_SIZE"], shuffle=False
    )

    # Create TFT model with specified hyperparameters
    model = TemporalFusionTransformer(
        num_static_features=num_static,
        num_known_features=num_known,
        num_observed_features=num_observed,
        hidden_size=config["HIDDEN_SIZE"],
        num_heads=config["NUM_HEADS"],
        dropout_rate=config["DROPOUT_RATE"],
        num_quantiles=config["NUM_QUANTILES"],
    )

    # Define optimizer with weight decay for regularization
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["LEARNING_RATE"],
        weight_decay=1e-5,  # Add L2 regularization
    )

    # Learning rate scheduler to reduce LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Training loop
    best_val_loss = float("inf")
    patience = 10  # Increased patience for early stopping
    patience_counter = 0
    epochs_completed = 0

    # Track losses for plotting
    train_losses = []
    val_losses = []

    print("\n=== Training Progress ===")
    print(f"{'Epoch':<6}{'Train Loss':<15}{'Val Loss':<15}{'LR':<10}")

    for epoch in range(config["EPOCHS"]):
        epochs_completed = epoch + 1

        # Training phase
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()

            # Get batch data
            past_data = batch["past_data"]
            static = batch["static"]
            target = batch["target"].unsqueeze(1)

            # Forward pass
            outputs = model(past_data, static)
            loss = quantile_loss(outputs["quantiles"], target)

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                past_data = batch["past_data"]
                static = batch["static"]
                target = batch["target"].unsqueeze(1)

                outputs = model(past_data, static)
                loss = quantile_loss(outputs["quantiles"], target)
                val_loss += loss.item()

        # Calculate average loss
        train_loss /= len(train_loader)
        val_loss /= len(test_loader)

        # Update learning rate scheduler
        scheduler.step(val_loss)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Store losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        # Print progress with better formatting
        print(f"{epoch+1:<6}{train_loss:.6f}      {val_loss:.6f}    {current_lr:.6f}")

        # Early stopping with more verbose output
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save model state
            best_model_state = model.state_dict().copy()
            print(f"  → New best model at epoch {epoch+1}")
        else:
            patience_counter += 1
            print(f"  → No improvement for {patience_counter}/{patience} epochs")
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Break if learning rate becomes too small
        if current_lr < 1e-6:
            print("Learning rate too small, stopping training")
            break

    # After training, plot the loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label="Training Loss")
    plt.plot(range(1, len(val_losses) + 1), val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss Curves")
    plt.legend()
    plt.grid(True)
    plt.savefig("tft_training_losses.png")
    print(f"Loss curves saved to tft_training_losses.png")

    # Load the best model state
    model.load_state_dict(best_model_state)

    return (
        best_val_loss,
        model,
        test_dataset,
        test_dates[config["SEQUENCE_LENGTH"] :],
        epochs_completed,
    )


def grid_search(data, static_features, dates, num_static, num_known, num_observed):
    """
    Perform grid search for TFT hyperparameters.
    Returns the best parameters, model, and evaluation results.
    """
    # Create all parameter combinations
    param_keys = list(PARAM_GRID.keys())
    param_values = list(PARAM_GRID.values())
    combinations = list(itertools.product(*param_values))

    # Initialize tracking variables
    best_val_loss = float("inf")
    best_params = None
    best_model = None
    best_test_dataset = None
    best_test_dates = None

    # Create results tracking dataframe
    results = []

    # Grid search
    print(f"Starting grid search with {len(combinations)} parameter combinations")

    # Create model directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    start_time = time.time()

    # Just try a few combinations for faster testing
    # You can uncomment this later to do a full grid search
    combinations = combinations[:3]

    for i, combination in enumerate(combinations):
        current_params = {param_keys[j]: combination[j] for j in range(len(param_keys))}

        # Update CONFIG with current parameters
        current_config = CONFIG.copy()
        for key, value in current_params.items():
            current_config[key] = value

        # Print current combination
        print(f"\nCombination {i+1}/{len(combinations)}:")
        for key, value in current_params.items():
            print(f"  {key}: {value}")

        try:
            # Train model with current parameters
            val_loss, model, test_dataset, test_dates, epochs = train_model_with_params(
                data,
                static_features,
                dates,
                num_static,
                num_known,
                num_observed,
                current_config,
            )

            # Track results
            result = {
                **current_params,
                "val_loss": val_loss,
                "epochs_completed": epochs,
                "combination_id": i + 1,
            }
            results.append(result)

            print(f"Validation loss: {val_loss:.6f} (epochs: {epochs})")

            # Check if this is the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_params = current_params.copy()
                best_model = model
                best_test_dataset = test_dataset
                best_test_dates = test_dates

                # Save the best model so far
                torch.save(model.state_dict(), "models/tft_model_best.pth")

                print(f"New best model! Validation loss: {best_val_loss:.6f}")

                # Save best parameters
                with open("models/best_params.json", "w") as f:
                    json.dump(best_params, f, indent=4)

        except Exception as e:
            print(f"Error with parameters {current_params}: {str(e)}")
            continue

        # Save interim results
        results_df = pd.DataFrame(results)
        results_df.to_csv("grid_search_results.csv", index=False)

    # Print final results
    elapsed_time = time.time() - start_time
    print(
        f"\nGrid search completed in {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)"
    )

    print("\nTop 5 parameter combinations:")
    results_df = pd.DataFrame(results)
    top_results = results_df.sort_values("val_loss").head(5)
    print(top_results)

    print("\nBest parameters:")
    for key, value in best_params.items():
        print(f"  {key}: {value}")
    print(f"Best validation loss: {best_val_loss:.6f}")

    # Update CONFIG with best parameters
    for key, value in best_params.items():
        CONFIG[key] = value

    return best_params, best_model, best_test_dataset, best_test_dates


def analyze_prediction_distribution(model, test_loader):
    """Analyze the distribution of predictions to check for flat predictions."""
    model.eval()

    all_p10 = []
    all_p50 = []
    all_p90 = []
    all_targets = []

    with torch.no_grad():
        for batch in test_loader:
            past_data = batch["past_data"]
            static = batch["static"]
            targets = batch["target"]

            outputs = model(past_data, static)

            # Get different quantiles
            p10 = outputs["p10"].squeeze().numpy()
            p50 = outputs["p50"].squeeze().numpy()
            p90 = outputs["p90"].squeeze().numpy()

            all_p10.extend(p10)
            all_p50.extend(p50)
            all_p90.extend(p90)
            all_targets.extend(targets.numpy())

    # Convert to numpy arrays
    p10_array = np.array(all_p10)
    p50_array = np.array(all_p50)
    p90_array = np.array(all_p90)
    targets_array = np.array(all_targets)

    # Analyze distributions
    print("\n=== Prediction Distribution Analysis ===")
    print(
        f"P10  - Mean: {np.mean(p10_array):.4f}, Std: {np.std(p10_array):.4f}, Min: {np.min(p10_array):.4f}, Max: {np.max(p10_array):.4f}"
    )
    print(
        f"P50  - Mean: {np.mean(p50_array):.4f}, Std: {np.std(p50_array):.4f}, Min: {np.min(p50_array):.4f}, Max: {np.max(p50_array):.4f}"
    )
    print(
        f"P90  - Mean: {np.mean(p90_array):.4f}, Std: {np.std(p90_array):.4f}, Min: {np.min(p90_array):.4f}, Max: {np.max(p90_array):.4f}"
    )
    print(
        f"Target - Mean: {np.mean(targets_array):.4f}, Std: {np.std(targets_array):.4f}"
    )

    # Check prediction width (P90-P10)
    pred_width = p90_array - p10_array
    print(
        f"Prediction Width (P90-P10) - Mean: {np.mean(pred_width):.4f}, Std: {np.std(pred_width):.4f}"
    )

    # Check if P50 is mainly predicting the mean
    target_mean = np.mean(targets_array)
    p50_mean = np.mean(p50_array)
    print(
        f"Difference between P50 mean and target mean: {abs(p50_mean - target_mean):.4f}"
    )

    # Check if predictions are too flat
    if np.std(p50_array) < 0.01:
        print(
            "WARNING: P50 predictions have very low variance. Model may be underfitting."
        )

        if abs(p50_mean - target_mean) < 0.1:
            print(
                "P50 is close to the target mean. Model is likely just predicting the average."
            )

        # Suggestions for fixing
        print("\nSuggestions to fix underfitting:")
        print("1. Increase model complexity (hidden_size, num_heads)")
        print("2. Train for more epochs")
        print("3. Reduce dropout rate")
        print("4. Check data preprocessing")
        print("5. Use feature selection to identify more informative variables")

    return p10_array, p50_array, p90_array, targets_array


def evaluate_model(model, test_dataset, test_dates):
    """Evaluate the model and generate predictions."""
    model.eval()

    # Create dataloader without shuffling to maintain order
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # First analyze prediction distribution to check for flat predictions
    p10_array, p50_array, p90_array, targets_array = analyze_prediction_distribution(
        model, test_loader
    )

    # Collect predictions and actual values
    all_predictions = []
    all_p10 = []
    all_p90 = []
    all_targets = []
    all_attention_weights = []
    all_model_uncertainty = []

    with torch.no_grad():
        for batch in test_loader:
            past_data = batch["past_data"]
            static = batch["static"]
            targets = batch["target"]

            # Get model outputs
            outputs = model(past_data, static)

            # Extract predictions and attention weights
            p50_predictions = outputs["p50"].squeeze().numpy()
            p10_predictions = outputs["p10"].squeeze().numpy()
            p90_predictions = outputs["p90"].squeeze().numpy()
            attention_weights = outputs["attention_weights"].numpy()

            # Calculate uncertainty as the difference between p90 and p10 predictions
            uncertainty = (
                p90_predictions - p10_predictions
            ) / 2  # Half the prediction interval width

            # Normalize uncertainty to [0, 1]
            max_uncertainty = np.max(uncertainty) if np.max(uncertainty) > 0 else 1
            normalized_uncertainty = uncertainty / max_uncertainty

            # Collect results
            all_predictions.extend(p50_predictions)
            all_p10.extend(p10_predictions)
            all_p90.extend(p90_predictions)
            all_targets.extend(targets.numpy())
            all_attention_weights.append(attention_weights)
            all_model_uncertainty.extend(normalized_uncertainty)

    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    p10 = np.array(all_p10)
    p90 = np.array(all_p90)
    actuals = np.array(all_targets)
    model_uncertainty = np.array(all_model_uncertainty)

    # Debug information - check if predictions are varying
    print("\nRaw prediction statistics:")
    print(f"P50 - Mean: {np.mean(predictions):.6f}, Std: {np.std(predictions):.6f}")
    print(f"P10 - Mean: {np.mean(p10):.6f}, Std: {np.std(p10):.6f}")
    print(f"P90 - Mean: {np.mean(p90):.6f}, Std: {np.std(p90):.6f}")

    # Check for flat predictions
    if np.std(predictions) < 0.01:
        print(
            "WARNING: Predictions have very low standard deviation, indicating flat predictions."
        )
        print("Try increasing model complexity or reducing regularization.")

    # Inverse transform predictions and actuals using target column from scaler
    target_scaler = CONFIG["target_scaler"]

    # Reshape for inverse transform
    temp_array = np.zeros((len(predictions), target_scaler.n_features_in_))
    temp_array[:, CONFIG["target_idx"]] = predictions

    p10_temp = np.zeros((len(p10), target_scaler.n_features_in_))
    p10_temp[:, CONFIG["target_idx"]] = p10

    p90_temp = np.zeros((len(p90), target_scaler.n_features_in_))
    p90_temp[:, CONFIG["target_idx"]] = p90

    target_temp = np.zeros((len(actuals), target_scaler.n_features_in_))
    target_temp[:, CONFIG["target_idx"]] = actuals

    # Inverse transform
    temp_array_inv = target_scaler.inverse_transform(temp_array)
    p10_inv = target_scaler.inverse_transform(p10_temp)
    p90_inv = target_scaler.inverse_transform(p90_temp)
    target_temp_inv = target_scaler.inverse_transform(target_temp)

    # Extract the target column
    predictions_inv = temp_array_inv[:, CONFIG["target_idx"]]
    p10_inv = p10_inv[:, CONFIG["target_idx"]]
    p90_inv = p90_inv[:, CONFIG["target_idx"]]
    actuals_inv = target_temp_inv[:, CONFIG["target_idx"]]

    # Debug the transformed predictions
    print("\nTransformed prediction statistics:")
    print(
        f"P50 - Mean: {np.mean(predictions_inv):.2f}, Std: {np.std(predictions_inv):.2f}"
    )
    print(f"P10 - Mean: {np.mean(p10_inv):.2f}, Std: {np.std(p10_inv):.2f}")
    print(f"P90 - Mean: {np.mean(p90_inv):.2f}, Std: {np.std(p90_inv):.2f}")
    print(f"Actuals - Mean: {np.mean(actuals_inv):.2f}, Std: {np.std(actuals_inv):.2f}")

    # Generate trading signals with confidence threshold from CONFIG
    signals, confidence = generate_trading_signals(
        predictions=predictions_inv,
        p10=p10_inv,
        p90=p90_inv,
        actuals=actuals_inv,
        model_uncertainty=model_uncertainty,
        confidence_threshold=CONFIG["CONFIDENCE_THRESHOLD"],
    )

    # Ensure we actually have some signals
    if np.sum(signals != 0) == 0:
        print("WARNING: No trading signals generated. Using price movement directly.")
        # Fallback approach - use price movements directly
        for i in range(len(predictions_inv) - 1):
            if (
                predictions_inv[i + 1] > actuals_inv[i] * 1.005
            ):  # 0.5% increase prediction
                signals[i] = 1  # Buy
                confidence[i] = CONFIG["CONFIDENCE_THRESHOLD"]  # Moderate confidence
            elif (
                predictions_inv[i + 1] < actuals_inv[i] * 0.995
            ):  # 0.5% decrease prediction
                signals[i] = -1  # Sell
                confidence[i] = CONFIG["CONFIDENCE_THRESHOLD"]  # Moderate confidence

    # Calculate RMSE
    rmse = np.sqrt(np.mean((predictions_inv - actuals_inv) ** 2))
    print(f"Test RMSE: {rmse:.2f}")
    print(
        f"Generated {np.sum(signals == 1)} buy signals and {np.sum(signals == -1)} sell signals"
    )

    # Create DataFrame with results
    results_df = pd.DataFrame(
        {
            "Date": test_dates,
            "Actual": actuals_inv,
            "Predicted": predictions_inv,
            "P10": p10_inv,
            "P90": p90_inv,
            "Signal": signals,
            "Confidence": confidence,
        }
    )

    # Load additional data for visualization
    temp_df = load_and_prepare_data()

    # Add technical indicators and sentiment
    results_df["EMA_8"] = np.nan
    results_df["SMA_200"] = np.nan
    results_df["Sentiment_Positive"] = np.nan
    results_df["Sentiment_Neutral"] = np.nan
    results_df["Sentiment_Negative"] = np.nan

    for i, date in enumerate(results_df["Date"]):
        if date in temp_df.index:
            if "EMA_8" in temp_df.columns:
                results_df.loc[i, "EMA_8"] = temp_df.loc[date, "EMA_8"]
            if "SMA_200" in temp_df.columns:
                results_df.loc[i, "SMA_200"] = temp_df.loc[date, "SMA_200"]
            if "sentiment_positive" in temp_df.columns:
                results_df.loc[i, "Sentiment_Positive"] = temp_df.loc[
                    date, "sentiment_positive"
                ]
            if "sentiment_neutral" in temp_df.columns:
                results_df.loc[i, "Sentiment_Neutral"] = temp_df.loc[
                    date, "sentiment_neutral"
                ]
            if "sentiment_negative" in temp_df.columns:
                results_df.loc[i, "Sentiment_Negative"] = temp_df.loc[
                    date, "sentiment_negative"
                ]

    # Plot results with confidence intervals
    plt.figure(figsize=(15, 7))
    plt.plot(results_df["Date"], results_df["Actual"], label="Actual", color="blue")
    plt.plot(
        results_df["Date"], results_df["Predicted"], label="Predicted", color="orange"
    )

    # Add confidence intervals
    plt.fill_between(
        results_df["Date"],
        results_df["P10"],
        results_df["P90"],
        color="orange",
        alpha=0.2,
        label="80% Confidence Interval",
    )

    if "EMA_8" in results_df.columns and not results_df["EMA_8"].isna().all():
        plt.plot(
            results_df["Date"],
            results_df["EMA_8"],
            label="8-day EMA",
            color="purple",
            linestyle="--",
            alpha=0.7,
        )

    if "SMA_200" in results_df.columns and not results_df["SMA_200"].isna().all():
        plt.plot(
            results_df["Date"],
            results_df["SMA_200"],
            label="200-day SMA",
            color="gray",
            linestyle="--",
            alpha=0.7,
        )

    # Plot buy/sell signals
    buy_signals = results_df["Signal"] == 1
    sell_signals = results_df["Signal"] == -1

    if buy_signals.any():
        plt.scatter(
            results_df["Date"][buy_signals],
            results_df["Actual"][buy_signals],
            color="green",
            marker="^",
            s=100,
            label="Buy Signal",
        )

    if sell_signals.any():
        plt.scatter(
            results_df["Date"][sell_signals],
            results_df["Actual"][sell_signals],
            color="red",
            marker="v",
            s=100,
            label="Sell Signal",
        )

    plt.title("AAPL Stock Price Prediction with Trading Signals (TFT Model)")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("tft_predictions_with_confidence.png")
    print("Predictions visualization saved to tft_predictions_with_confidence.png")

    # Save predictions to CSV
    results_df.to_csv("tft_predictions.csv", index=False)
    print("Predictions saved to tft_predictions.csv")

    return results_df


def generate_trading_signals(
    predictions, p10, p90, actuals, model_uncertainty, confidence_threshold=0.6
):
    """
    Generate trading signals based on TFT predictions, uncertainty and trend direction.

    Args:
        predictions: Array of price predictions (P50)
        p10: Array of lower bound predictions (P10)
        p90: Array of upper bound predictions (P90)
        actuals: Array of actual prices
        model_uncertainty: Array of model uncertainty values (from quantile differences)
        confidence_threshold: Minimum confidence required to generate signals

    Returns:
        signals: Array of trading signals (1=buy, -1=sell, 0=no action)
        confidence: Array of confidence scores for each signal
    """
    # Calculate prediction errors
    errors = np.abs(predictions - actuals)
    max_error = np.max(errors) if np.max(errors) > 0 else 1.0

    # Calculate confidence from normalized error (inverse relationship)
    confidence_from_error = 1 - (errors / max_error)

    # Adjust confidence based on model uncertainty
    confidence = confidence_from_error * (
        1 - model_uncertainty * 0.7
    )  # Increase uncertainty impact

    # Initialize signals
    signals = np.zeros(len(predictions))

    # Calculate price movement predictions
    # Look ahead prediction: difference between tomorrow's predicted price and today's actual price
    price_changes = np.zeros(len(predictions))
    for i in range(len(predictions) - 1):
        price_changes[i] = predictions[i + 1] - actuals[i]
    price_changes[-1] = predictions[-1] - actuals[-1]  # For the last point

    # Normalize price changes
    price_change_pct = price_changes / actuals

    # Calculate prediction width (P90 - P10) relative to price
    # Narrow width means higher confidence
    pred_width_pct = (p90 - p10) / actuals

    # Reduce confidence when prediction interval is too wide
    confidence = confidence * (1 - np.minimum(pred_width_pct, 1) * 0.5)

    # Generate signals based on predicted TREND rather than just price differences
    high_confidence = confidence >= confidence_threshold

    # Stronger thresholds for meaningful signals (0.5% change)
    buy_threshold = 0.005  # 0.5% predicted increase
    sell_threshold = -0.005  # 0.5% predicted decrease

    # Buy signal: tomorrow's price will be significantly higher than today's
    signals[high_confidence & (price_change_pct > buy_threshold)] = 1

    # Sell signal: tomorrow's price will be significantly lower than today's
    signals[high_confidence & (price_change_pct < sell_threshold)] = -1

    # Calculate final confidence for the signals
    signal_confidence = confidence.copy()

    # Boost confidence for stronger price movements
    for i in range(len(signals)):
        if signals[i] != 0:
            # Boost confidence based on strength of predicted movement
            move_strength = min(abs(price_change_pct[i]) / 0.01, 1.0)  # Cap at 1.0
            signal_confidence[i] = min(confidence[i] * (1 + move_strength * 0.5), 1.0)

    # Set confidence to 0 for no-signal periods
    signal_confidence[signals == 0] = 0

    return signals, signal_confidence


def main():
    # Create model directory if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Load data
    df = load_and_prepare_data()

    # Prepare data for TFT
    data, static_features, dates, num_static, num_known, num_observed = (
        prepare_data_for_tft(df)
    )

    # Check if data preparation was successful
    if data is None:
        print("Data preparation failed. Please check your input data.")
        return

    print(
        f"Data prepared with {num_static} static features, {num_known} known features, and {num_observed} observed features"
    )

    # Skip grid search for initial testing - just use default config
    print("\nSkipping grid search for initial testing...")
    config = CONFIG

    # Run initial training to diagnose issues
    best_val_loss, model, test_dataset, test_dates, epochs_completed = (
        train_model_with_params(
            data, static_features, dates, num_static, num_known, num_observed, config
        )
    )

    # Create test dataloader
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Analyze prediction distribution to check for flat predictions
    _ = analyze_prediction_distribution(model, test_loader)

    # Evaluate the model
    print("\nEvaluating model...")
    results_df = evaluate_model(model, test_dataset, test_dates)

    print("\nTrading simulation will be executed using tft_trading_simulation.py")
    print("Model saved to models/tft_model_best.pth")


if __name__ == "__main__":
    main()

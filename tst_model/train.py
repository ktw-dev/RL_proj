# tst_model/train.py: Script for pre-training the TST model.
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import MinMaxScaler
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import os
import sys
from datetime import datetime

# Add project root to sys.path to allow for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from tst_model.model import TSTModel # Assuming model.py is in the same directory or accessible
# from config import settings # If specific paths or configs are needed

# Configuration (Ideally, this would come from a config file or command-line arguments)
DEFAULT_MODEL_CONFIG = {
    'input_size': 87,  # 80 TA features + 7 news sentiment features
    'prediction_length': 10, # How many steps to predict into the future
    'context_length': 60,    # How many past steps to use as context (back to 60)
    'n_layer': 4,            # Number of transformer layers
    'n_head': 8,             # Number of attention heads
    'd_model': 128,          # Dimensionality of the model
    'rl_state_size': 256,    # Desired size of the RL agent's state vector
}

TRAIN_CONFIG = {
    'data_path': os.path.join(PROJECT_ROOT, 'all_tickers_historical.csv'), # Updated path to match actual filename  
    'output_dir': os.path.join(PROJECT_ROOT, 'tst_model_output'),
    'batch_size': 32,
    'epochs': 100,  # Increased epochs since we'll have better early stopping
    'learning_rate': 5e-4,  # Reduced learning rate for more stable training
    'weight_decay': 0.01,
    'patience_early_stopping': 15,  # Increased patience for more stable training
    'min_delta': 1e-6,  # Minimum improvement to count as progress
    'validation_split_ratio': 0.2,
    'random_seed': 42,
    'warmup_steps_ratio': 0.1,  # 10% of total steps for warmup
    'scheduler_patience': 5,  # For ReduceLROnPlateau scheduler
    'factor': 0.5,  # Factor to reduce LR when plateau is detected
}

NEWS_FEATURE_COLS = [
    'avg_sentiment_positive', 'avg_sentiment_negative', 'avg_sentiment_neutral',
    'news_count', 'weekend_effect_positive', 'weekend_effect_negative', 'weekend_effect_neutral'
]

def add_synthetic_news_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add synthetic neutral news features directly to the DataFrame."""
    print("Adding synthetic neutral news features...")
    
    # Define neutral news feature values
    news_features = {
        'avg_sentiment_positive': 0.0,
        'avg_sentiment_negative': 0.0,
        'avg_sentiment_neutral': 1.0,
        'news_count': 0,
        'weekend_effect_positive': 0.0,
        'weekend_effect_negative': 0.0,
        'weekend_effect_neutral': 0.0,
    }
    
    # Add each news feature column with default neutral values
    for col, default_value in news_features.items():
        df[col] = default_value
    
    print(f"Added {len(news_features)} synthetic news features")
    return df

def create_sequences(data_df: pd.DataFrame, context_length: int, prediction_length: int, target_cols_indices=None):
    """
    Creates sequences of (past_values, future_values) for TST model training.
    Data is processed per ticker.
    target_cols_indices: List of column indices to use for future_values. If None, all columns are used.
    """
    all_past_sequences = []
    all_future_sequences = []
    
    # Ensure data is sorted by Date within each Ticker group
    data_df = data_df.sort_index(level=['Ticker', 'Date'])
    
    # Get all non-index columns as features
    feature_columns = [col for col in data_df.columns]

    for ticker, group in data_df.groupby(level='Ticker'):
        ticker_data = group[feature_columns].values # Get as numpy array
        if len(ticker_data) < context_length + prediction_length:
            print(f"Skipping ticker {ticker} due to insufficient data for sequences (has {len(ticker_data)} rows, need {context_length + prediction_length})")
            continue

        for i in range(len(ticker_data) - context_length - prediction_length + 1):
            past_seq = ticker_data[i : i + context_length]
            future_seq = ticker_data[i + context_length : i + context_length + prediction_length]
            
            all_past_sequences.append(past_seq)
            if target_cols_indices is not None:
                all_future_sequences.append(future_seq[:, target_cols_indices])
            else:
                all_future_sequences.append(future_seq)
                
    if not all_past_sequences: # Check if any sequences were created
        print("Error: No sequences were created from any ticker")
        return None, None

    print(f"Created sequences from {len(set([ticker for ticker, _ in data_df.groupby(level='Ticker')]))} tickers")
    return torch.FloatTensor(np.array(all_past_sequences)), torch.FloatTensor(np.array(all_future_sequences))

def train_tst_model(model_config: dict, train_config: dict):
    """
    Main training loop for the TST model.
    """
    print("Starting TST model training...")
    np.random.seed(train_config['random_seed'])
    torch.manual_seed(train_config['random_seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(train_config['random_seed'])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load TA data
    print(f"Loading historical TA data from {train_config['data_path']}")
    try:
        ta_history_df = pd.read_csv(train_config['data_path'])
    except FileNotFoundError:
        print(f"Error: TA data file not found at {train_config['data_path']}. Exiting.")
        return

    # Convert Date to datetime and set up MultiIndex - Ticker first for proper grouping
    ta_history_df['Date'] = pd.to_datetime(ta_history_df['Date'])
    if not isinstance(ta_history_df.index, pd.MultiIndex):
        if 'Ticker' in ta_history_df.columns:
            # Set Ticker first, then Date for proper grouping by ticker
            ta_history_df.set_index(['Ticker', 'Date'], inplace=True)
        else:
            print("Error: 'Ticker' column missing in CSV for MultiIndex setup.")
            return
    ta_history_df.sort_index(inplace=True)
    print(f"Loaded TA data. Shape: {ta_history_df.shape}")
    print(f"Tickers found: {sorted(ta_history_df.index.get_level_values('Ticker').unique().tolist())}")

    # 2. Add synthetic neutral news features for training compatibility 
    combined_features_df = ta_history_df.copy()
    
    # Get only numeric columns (exclude Date and Ticker)
    numeric_cols = combined_features_df.select_dtypes(include=['number']).columns.tolist()
    print(f"Found {len(numeric_cols)} numeric TA features: {numeric_cols[:5]}...{numeric_cols[-5:]}")
    
    # Add synthetic neutral news sentiment features for training
    print("Adding synthetic neutral news sentiment features for training...")
    news_features = {
        'avg_sentiment_positive': 0.0,
        'avg_sentiment_negative': 0.0,
        'avg_sentiment_neutral': 1.0,
        'news_count': 0,
        'weekend_effect_positive': 0.0,
        'weekend_effect_negative': 0.0,
        'weekend_effect_neutral': 0.0,
    }
    
    # Add each news feature column with default neutral values
    for col, default_value in news_features.items():
        combined_features_df[col] = default_value
    
    # Update feature list to include news features
    all_feature_cols = numeric_cols + list(news_features.keys())
    print(f"Total features after adding news sentiment: {len(all_feature_cols)} (80 TA + 7 news)")
    
    # Update input_size in model_config based on actual number of features
    model_config['input_size'] = len(all_feature_cols)
    print(f"Updated model_config input_size to: {model_config['input_size']}")
    
    # Keep only the feature columns for training
    combined_features_df = combined_features_df[all_feature_cols]
    print(f"Final features shape: {combined_features_df.shape}")

    # 3. Feature Scaling (per-ticker for better performance)
    print("Scaling features per ticker...")
    feature_cols_to_scale = all_feature_cols  # Use all features including news sentiment
    scalers = {} # To store scalers for each ticker 
    scaled_df_list = []

    for ticker, group in combined_features_df.groupby(level='Ticker'):
        scaler = MinMaxScaler()
        scaled_group_values = scaler.fit_transform(group[feature_cols_to_scale])
        scaled_group_df = pd.DataFrame(scaled_group_values, index=group.index, columns=feature_cols_to_scale)
        scalers[ticker] = scaler # Save scaler for potential future use
        scaled_df_list.append(scaled_group_df)
        print(f"Scaled {ticker}: {len(group)} samples")
    
    if not scaled_df_list:
        print("Error: No data after grouping by ticker for scaling. Exiting.")
        return

    scaled_features_df = pd.concat(scaled_df_list)
    scaled_features_df.sort_index(inplace=True)
    print(f"Features scaled. Final shape: {scaled_features_df.shape}")

    # 4. Create sequences
    print("Creating sequences for TST model...")
    
    past_sequences, future_sequences = create_sequences(
        scaled_features_df,
        context_length=model_config['context_length'],
        prediction_length=model_config['prediction_length']
    )

    if past_sequences is None or future_sequences is None or past_sequences.nelement() == 0:
        print("Error: No sequences were created. Check data length and sequence parameters. Exiting.")
        return
    print(f"Created {len(past_sequences)} sequences.")
    print(f"Past sequences shape: {past_sequences.shape}, Future sequences shape: {future_sequences.shape}")

    # 5. Create DataLoader
    dataset = TensorDataset(past_sequences, future_sequences)
    val_size = int(len(dataset) * train_config['validation_split_ratio'])
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=train_config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)
    print(f"Train loader: {len(train_loader)} batches, Val loader: {len(val_loader)} batches")

    # 6. Initialize Model, Optimizer, Scheduler
    print("Initializing TST model...")
    model = TSTModel(config_dict=model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model initialized with {total_params:,} parameters")
    
    optimizer = AdamW(model.parameters(), lr=train_config['learning_rate'], weight_decay=train_config['weight_decay'])
    
    # Use both warmup scheduler and plateau scheduler for better stability
    total_steps = len(train_loader) * train_config['epochs']
    warmup_steps = int(total_steps * train_config['warmup_steps_ratio'])
    warmup_scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=warmup_steps, 
        num_training_steps=total_steps
    )
    
    # Add ReduceLROnPlateau for additional stability
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    plateau_scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=train_config['factor'], 
        patience=train_config['scheduler_patience'],
        min_lr=1e-7
    )

    # 7. Training Loop with Improved Early Stopping
    print("Starting training loop...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    val_loss_history = []  # For smoothing validation loss
    os.makedirs(train_config['output_dir'], exist_ok=True)
    model_path = os.path.join(train_config['output_dir'], f"tst_model_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt")

    print(f"Early stopping patience: {train_config['patience_early_stopping']}")
    print(f"Minimum improvement delta: {train_config['min_delta']}")

    for epoch in range(train_config['epochs']):
        model.train()
        total_train_loss = 0
        num_batches = 0
        
        for batch_idx, (past_vals, future_vals) in enumerate(train_loader):
            past_vals, future_vals = past_vals.to(device), future_vals.to(device)
            
            optimizer.zero_grad()
            
            # For TimeSeriesTransformerForPrediction, provide future_values for loss calculation
            outputs = model(past_values=past_vals, future_values=future_vals)
            loss = outputs.loss
            
            if loss is None:
                print(f"Epoch {epoch+1} Batch {batch_idx+1}: Loss is None. Skipping backward pass.")
                continue

            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Only step warmup scheduler during warmup phase
            if epoch * len(train_loader) + batch_idx < warmup_steps:
                warmup_scheduler.step()
            
            total_train_loss += loss.item()
            num_batches += 1

            if (batch_idx + 1) % 20 == 0:  # Less frequent logging
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch {epoch+1}/{train_config['epochs']}, Batch {batch_idx+1}/{len(train_loader)}, Train Loss: {loss.item():.6f}, LR: {current_lr:.2e}")

        avg_train_loss = total_train_loss / num_batches if num_batches > 0 else float('inf')

        # Validation
        model.eval()
        total_val_loss = 0
        num_val_batches = 0
        
        with torch.no_grad():
            for past_vals, future_vals in val_loader:
                past_vals, future_vals = past_vals.to(device), future_vals.to(device)
                
                # Force model into training mode temporarily for loss calculation during validation
                model.train()
                outputs = model(past_values=past_vals, future_values=future_vals)
                model.eval()
                
                val_loss = outputs.loss
                if val_loss is not None:
                    total_val_loss += val_loss.item()
                    num_val_batches += 1
                else:
                    print("Warning: Validation loss is None during evaluation.")
        
        avg_val_loss = total_val_loss / num_val_batches if num_val_batches > 0 else float('inf')
        val_loss_history.append(avg_val_loss)
        
        # Smooth validation loss over last 3 epochs for more stable early stopping
        if len(val_loss_history) >= 3:
            smoothed_val_loss = np.mean(val_loss_history[-3:])
        else:
            smoothed_val_loss = avg_val_loss
            
        # Step plateau scheduler with smoothed validation loss
        plateau_scheduler.step(smoothed_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} Summary: Avg Train Loss: {avg_train_loss:.6f}, Avg Val Loss: {avg_val_loss:.6f}, Smoothed Val Loss: {smoothed_val_loss:.6f}, LR: {current_lr:.2e}")

        # Improved early stopping with min_delta
        val_loss_improvement = best_val_loss - smoothed_val_loss
        
        if val_loss_improvement > train_config['min_delta']:
            best_val_loss = smoothed_val_loss
            torch.save(model.state_dict(), model_path)
            print(f"Validation loss improved by {val_loss_improvement:.6f}. Saved model to {model_path}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if val_loss_improvement <= 0:
                print(f"Validation loss did not improve for {epochs_no_improve} epoch(s).")
            else:
                print(f"Validation loss improved by {val_loss_improvement:.6f} (less than min_delta {train_config['min_delta']:.6f}) for {epochs_no_improve} epoch(s).")
            
            if epochs_no_improve >= train_config['patience_early_stopping']:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break
                
    print("Training finished.")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Total epochs trained: {epoch+1}")
    print(f"Final learning rate: {optimizer.param_groups[0]['lr']:.2e}")
    print(f"Model saved to {model_path}")
    
    # Load best model for final validation
    model.load_state_dict(torch.load(model_path))
    print("Loaded best model for final evaluation")

if __name__ == '__main__':
    # Check if the actual data file exists, create dummy if not
    data_file_path = TRAIN_CONFIG['data_path']
    if not os.path.exists(data_file_path):
        # Try alternative filenames
        alternative_paths = [
            os.path.join(PROJECT_ROOT, 'all_tickers_historical_features.csv'),
            os.path.join(PROJECT_ROOT, 'all_tickers_historical.csv')
        ]
        
        found_file = None
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                found_file = alt_path
                break
        
        if found_file:
            print(f"Using found data file: {found_file}")
            TRAIN_CONFIG['data_path'] = found_file
        else:
            print(f"No data file found. Creating dummy data for testing at {data_file_path}")
            # Create dummy data with proper structure
            num_tickers = 3
            num_days_per_ticker = DEFAULT_MODEL_CONFIG['context_length'] + DEFAULT_MODEL_CONFIG['prediction_length'] + 20
            num_ta_features = 81
            
            dates = pd.date_range(start='2023-01-01', periods=num_days_per_ticker, freq='D')
            df_list = []
            
            for i in range(num_tickers):
                ticker_name = f"TICKER{i+1}"
                temp_df = pd.DataFrame(np.random.rand(num_days_per_ticker, num_ta_features), 
                                     columns=[f'TA_feature_{j}' for j in range(num_ta_features)])
                temp_df['Date'] = dates
                temp_df['Ticker'] = ticker_name
                df_list.append(temp_df)
            
            dummy_df = pd.concat(df_list, ignore_index=True)
            # Save without setting index for CSV compatibility
            dummy_df.to_csv(data_file_path, index=False)
            print(f"Created dummy data file at {data_file_path}")

    train_tst_model(model_config=DEFAULT_MODEL_CONFIG.copy(), train_config=TRAIN_CONFIG.copy()) 
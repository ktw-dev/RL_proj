# tst_model/predict.py: Script for using the trained TST model for inference.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import os
import sys
import glob
from datetime import datetime, timedelta
import argparse

# Add project root to sys.path to allow for absolute imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from tst_model.model import TSTModel
# from feature_engineering.feature_combiner import align_and_combine_features  # Not used in train.py approach

# Configuration matching train.py
DEFAULT_MODEL_CONFIG = {
    'input_size': 87,  # Set by data: 80 TA + 7 News (matching train.py)
    'prediction_length': 10, # How many steps to predict into the future
    'context_length': 60,    # How many past steps to use as context
    'n_layer': 4,            # Number of transformer layers (matching train.py)
    'n_head': 8,             # Number of attention heads (matching train.py)
    'd_model': 128,          # Dimensionality of the model
    'rl_state_size': 256,    # Desired size of the RL agent's state vector
    'distribution_output': "normal", 
    'loss': "nll",             
    'num_parallel_samples': 100
}

PREDICT_CONFIG = {
    'model_dir': os.path.join(PROJECT_ROOT, 'tst_model_output'),
    'data_path': os.path.join(PROJECT_ROOT, 'all_tickers_historical_features.csv'),  # Will fallback to alternatives
    'output_dir': os.path.join(PROJECT_ROOT, 'tst_predictions'),
    'batch_size': 32
}

NEWS_FEATURE_COLS = [
    'avg_sentiment_positive', 'avg_sentiment_negative', 'avg_sentiment_neutral',
    'news_count', 'weekend_effect_positive', 'weekend_effect_negative', 'weekend_effect_neutral'
]

# create_neutral_news_df function removed - using train.py approach of adding features directly

def load_latest_model(model_dir: str, model_config: dict, device: torch.device):
    """
    Load the most recent trained TST model from the model directory.
    
    Args:
        model_dir (str): Directory containing saved models
        model_config (dict): Model configuration dictionary
        device (torch.device): Device to load the model on
        
    Returns:
        TSTModel: Loaded model in evaluation mode
        str: Path to the loaded model file
    """
    # Find the most recent model file
    model_pattern = os.path.join(model_dir, "tst_model_best_*.pt")
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        raise FileNotFoundError(f"No trained models found in {model_dir}. Expected pattern: tst_model_best_*.pt")
    
    # Sort by modification time (most recent first)
    latest_model_path = max(model_files, key=os.path.getmtime)
    
    print(f"Loading model from: {latest_model_path}")
    
    # Initialize model
    model = TSTModel(config_dict=model_config).to(device)
    
    # Load trained weights
    checkpoint = torch.load(latest_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    
    # Set to evaluation mode
    model.eval()
    
    print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, latest_model_path

def prepare_data_for_prediction(data_path: str, target_ticker: str = None, context_length: int = 60):
    """
    Prepare and scale data for TST model prediction using train.py approach.
    
    Args:
        data_path (str): Path to the historical data CSV
        target_ticker (str): Specific ticker to predict (if None, process all)
        context_length (int): Number of past days needed for context
        
    Returns:
        dict: Contains processed data, scalers, and metadata
    """
    print(f"Loading historical data from: {data_path}")
    
    # Check if data file exists, try alternatives like train.py
    if not os.path.exists(data_path):
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
            data_path = found_file
        else:
            raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load TA data
    ta_history_df = pd.read_csv(data_path)
    
    # Setup DataFrame structure (same as train.py)
    ta_history_df['Date'] = pd.to_datetime(ta_history_df['Date'])
    if not isinstance(ta_history_df.index, pd.MultiIndex):
        if 'Ticker' in ta_history_df.columns:
            # Set Ticker first, then Date for proper grouping by ticker (same as train.py)
            ta_history_df.set_index(['Ticker', 'Date'], inplace=True)
        else:
            raise ValueError("'Ticker' column missing in CSV for MultiIndex setup.")
    ta_history_df.sort_index(inplace=True)
    
    # Filter for specific ticker if requested
    if target_ticker:
        if target_ticker not in ta_history_df.index.get_level_values('Ticker'):
            raise ValueError(f"Ticker '{target_ticker}' not found in data")
        ta_history_df = ta_history_df.xs(target_ticker, level='Ticker', drop_level=False)
    
    print(f"Loaded TA data. Shape: {ta_history_df.shape}")
    print(f"Tickers found: {sorted(ta_history_df.index.get_level_values('Ticker').unique().tolist())}")
    
    # Get only numeric columns (exclude Date and Ticker) - same as train.py
    combined_features_df = ta_history_df.copy()
    numeric_cols = combined_features_df.select_dtypes(include=['number']).columns.tolist()
    print(f"Found {len(numeric_cols)} numeric TA features")
    
    # Add synthetic neutral news sentiment features for training (same as train.py)
    print("Adding synthetic neutral news sentiment features...")
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
    print(f"Total features after adding news sentiment: {len(all_feature_cols)} ({len(numeric_cols)} TA + 7 news)")
    
    # Keep only the feature columns for prediction
    combined_features_df = combined_features_df[all_feature_cols]
    print(f"Final features shape: {combined_features_df.shape}")
    
    # Feature scaling per ticker (same as train.py)
    print("Scaling features per ticker...")
    scalers = {}
    scaled_df_list = []
    
    for ticker, group in combined_features_df.groupby(level='Ticker'):
        if len(group) < context_length:
            print(f"Warning: Ticker {ticker} has insufficient data ({len(group)} < {context_length}). Skipping.")
            continue
            
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(group[all_feature_cols])
        scaled_df = pd.DataFrame(scaled_values, index=group.index, columns=all_feature_cols)
        
        scalers[ticker] = scaler
        scaled_df_list.append(scaled_df)
        print(f"Scaled {ticker}: {len(group)} samples")
    
    if not scaled_df_list:
        raise ValueError("No valid ticker data after scaling")
    
    scaled_features_df = pd.concat(scaled_df_list).sort_index()
    print(f"Features scaled. Final shape: {scaled_features_df.shape}")
    
    return {
        'scaled_data': scaled_features_df,
        'scalers': scalers,
        'feature_columns': all_feature_cols,
        'raw_data': combined_features_df
    }

def create_prediction_sequences(scaled_data: pd.DataFrame, context_length: int):
    """
    Create sequences for prediction (only past values needed).
    
    Args:
        scaled_data (pd.DataFrame): Scaled feature data
        context_length (int): Length of context window
        
    Returns:
        dict: Contains sequences and metadata for each ticker
    """
    prediction_data = {}
    
    for ticker, group in scaled_data.groupby(level='Ticker'):
        ticker_data = group.values
        
        if len(ticker_data) < context_length:
            print(f"Warning: Ticker {ticker} insufficient data for prediction. Skipping.")
            continue
        
        # Get the most recent context_length data points for prediction
        latest_sequence = ticker_data[-context_length:]  # Shape: (context_length, n_features)
        latest_dates = group.index.get_level_values('Date')[-context_length:]
        
        prediction_data[ticker] = {
            'sequence': torch.FloatTensor(latest_sequence).unsqueeze(0),  # Add batch dimension
            'dates': latest_dates,
            'last_date': latest_dates[-1]
        }
        
        print(f"Prepared prediction sequence for {ticker}: {latest_sequence.shape}")
    
    return prediction_data

def predict_with_tst_model(model: TSTModel, prediction_data: dict, device: torch.device, mode: str = 'rl_state'):
    """
    Perform prediction using the TST model.
    
    Args:
        model (TSTModel): Trained TST model
        prediction_data (dict): Prepared prediction sequences
        device (torch.device): Device for computation
        mode (str): 'rl_state' for RL state vector, 'forecast' for future predictions
        
    Returns:
        dict: Prediction results for each ticker
    """
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for ticker, data in prediction_data.items():
            print(f"Predicting for ticker: {ticker}")
            
            sequence = data['sequence'].to(device)  # Shape: (1, context_length, n_features)
            
            if mode == 'rl_state':
                # Get RL state vector for reinforcement learning agent
                rl_state = model(past_values=sequence)  # Shape: (1, rl_state_size)
                predictions[ticker] = {
                    'rl_state': rl_state.cpu().numpy().squeeze(),  # Shape: (rl_state_size,)
                    'last_date': data['last_date'],
                    'prediction_type': 'rl_state'
                }
                print(f"  RL State shape: {rl_state.shape}")
                
            elif mode == 'forecast':
                # Use the model's predict_future method for forecasting
                forecast = model.predict_future(sequence)  # Shape: (1, prediction_length, n_features)
                
                predictions[ticker] = {
                    'forecast': forecast.cpu().numpy().squeeze(),  # Shape: (prediction_length, n_features)
                    'last_date': data['last_date'],
                    'prediction_type': 'forecast'
                }
                print(f"  Forecast shape: {forecast.shape}")
                
    return predictions

def save_predictions(predictions: dict, output_dir: str, model_path: str):
    """
    Save prediction results to files.
    
    Args:
        predictions (dict): Prediction results
        output_dir (str): Output directory
        model_path (str): Path to the model used for predictions
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save metadata
    metadata = {
        'timestamp': timestamp,
        'model_path': model_path,
        'num_tickers': len(predictions),
        'tickers': list(predictions.keys())
    }
    
    # Save individual ticker predictions
    for ticker, pred_data in predictions.items():
        if pred_data['prediction_type'] == 'rl_state':
            # Save RL state vector
            output_file = os.path.join(output_dir, f"{ticker}_rl_state_{timestamp}.npy")
            np.save(output_file, pred_data['rl_state'])
            print(f"Saved RL state for {ticker}: {output_file}")
            
        elif pred_data['prediction_type'] == 'forecast':
            # Save forecast as CSV
            forecast_df = pd.DataFrame(pred_data['forecast'])
            forecast_df.index.name = 'prediction_day'
            output_file = os.path.join(output_dir, f"{ticker}_forecast_{timestamp}.csv")
            forecast_df.to_csv(output_file)
            print(f"Saved forecast for {ticker}: {output_file}")
    
    # Save summary
    summary_file = os.path.join(output_dir, f"prediction_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"TST Model Prediction Summary\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Number of tickers: {len(predictions)}\n")
        f.write(f"Tickers: {', '.join(predictions.keys())}\n\n")
        
        for ticker, pred_data in predictions.items():
            f.write(f"{ticker}:\n")
            f.write(f"  Last data date: {pred_data['last_date']}\n")
            f.write(f"  Prediction type: {pred_data['prediction_type']}\n")
            if pred_data['prediction_type'] == 'rl_state':
                f.write(f"  RL state size: {len(pred_data['rl_state'])}\n")
                f.write(f"  RL state mean: {np.mean(pred_data['rl_state']):.4f}\n")
                f.write(f"  RL state std: {np.std(pred_data['rl_state']):.4f}\n")
            f.write("\n")
    
    print(f"Saved prediction summary: {summary_file}")

def main():
    """Main prediction function with command-line interface."""
    parser = argparse.ArgumentParser(description='TST Model Prediction')
    parser.add_argument('--ticker', type=str, help='Specific ticker to predict (default: all tickers)')
    parser.add_argument('--mode', type=str, choices=['rl_state', 'forecast'], default='rl_state',
                       help='Prediction mode: rl_state or forecast')
    parser.add_argument('--model_dir', type=str, default=PREDICT_CONFIG['model_dir'],
                       help='Directory containing trained models')
    parser.add_argument('--data_path', type=str, default=PREDICT_CONFIG['data_path'],
                       help='Path to historical data CSV')
    parser.add_argument('--output_dir', type=str, default=PREDICT_CONFIG['output_dir'],
                       help='Output directory for predictions')
    
    args = parser.parse_args()
    
    print("=== TST Model Prediction ===")
    print(f"Target ticker: {args.ticker or 'All tickers'}")
    print(f"Prediction mode: {args.mode}")
    print(f"Model directory: {args.model_dir}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # Prepare data first to determine actual feature count
        data_info = prepare_data_for_prediction(
            args.data_path, 
            target_ticker=args.ticker,
            context_length=DEFAULT_MODEL_CONFIG['context_length']
        )
        
        # Update model config with actual input size (like train.py)
        model_config = DEFAULT_MODEL_CONFIG.copy()
        actual_input_size = len(data_info['feature_columns'])
        model_config['input_size'] = actual_input_size
        print(f"Updated model_config input_size to: {model_config['input_size']} (from data)")
        
        # Load model with updated config
        model, model_path = load_latest_model(args.model_dir, model_config, device)
        
        # Create prediction sequences
        prediction_data = create_prediction_sequences(
            data_info['scaled_data'],
            model_config['context_length']
        )
        
        if not prediction_data:
            print("No valid data for prediction. Exiting.")
            return
        
        # Perform predictions
        predictions = predict_with_tst_model(model, prediction_data, device, mode=args.mode)
        
        # Save results
        save_predictions(predictions, args.output_dir, model_path)
        
        print("=== Prediction Complete ===")
        print(f"Processed {len(predictions)} ticker(s)")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 
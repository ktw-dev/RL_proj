# inference_example.py: Example of how to use the trained TST model with real news sentiment features
import pandas as pd
import numpy as np
import torch
from datetime import datetime, timezone
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from tst_model.model import TSTModel
from data_collection.news_analyzer import fetch_and_analyze_recent_news  
from feature_engineering.news_processor import aggregate_daily_sentiment_features

def load_trained_model(model_path, config):
    """Load the trained TST model."""
    model = TSTModel(config)
    model.load_state_dict(torch.load(model_path, map_location='cpu'))
    model.eval()
    return model

def get_latest_ta_features(ticker, num_days=60):
    """
    Get the latest 60 days of TA features for a ticker.
    In real implementation, this would fetch from your TA data source.
    For now, we'll simulate with the CSV data.
    """
    # Load the historical data CSV
    df = pd.read_csv('all_tickers_historical_features.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Filter for the specific ticker and get the latest data
    ticker_data = df[df['Ticker'] == ticker].copy()
    ticker_data = ticker_data.sort_values('Date').tail(num_days)
    
    # Get numeric TA features (80 features)
    numeric_cols = ticker_data.select_dtypes(include=['number']).columns.tolist()
    ta_features = ticker_data[numeric_cols].values  # Shape: (num_days, 80)
    
    return ta_features, ticker_data['Date'].tolist()

def get_news_sentiment_features(ticker, analysis_date=None):
    """
    Get real news sentiment features for a ticker.
    Returns 7 sentiment features as a vector.
    """
    if analysis_date is None:
        analysis_date = datetime.now(timezone.utc)
    
    try:
        # Fetch and analyze recent news
        print(f"Fetching news sentiment for {ticker}...")
        news_df = fetch_and_analyze_recent_news(ticker, analysis_date)
        
        if news_df.empty:
            print(f"No news found for {ticker}, using neutral sentiment")
            return np.array([0.0, 0.0, 1.0, 0, 0.0, 0.0, 0.0])  # Neutral sentiment
        
        # Process news into daily sentiment features
        sentiment_features_df = aggregate_daily_sentiment_features(news_df, ticker)
        
        if sentiment_features_df.empty:
            print(f"No sentiment features processed for {ticker}, using neutral sentiment")
            return np.array([0.0, 0.0, 1.0, 0, 0.0, 0.0, 0.0])  # Neutral sentiment
        
        # Get the most recent day's sentiment features
        latest_sentiment = sentiment_features_df.iloc[-1]
        sentiment_vector = np.array([
            latest_sentiment['avg_sentiment_positive'],
            latest_sentiment['avg_sentiment_negative'], 
            latest_sentiment['avg_sentiment_neutral'],
            latest_sentiment['news_count'],
            latest_sentiment['weekend_effect_positive'],
            latest_sentiment['weekend_effect_negative'],
            latest_sentiment['weekend_effect_neutral']
        ])
        
        print(f"News sentiment for {ticker}: {sentiment_vector}")
        return sentiment_vector
        
    except Exception as e:
        print(f"Error getting news sentiment for {ticker}: {e}")
        print("Using neutral sentiment as fallback")
        return np.array([0.0, 0.0, 1.0, 0, 0.0, 0.0, 0.0])  # Neutral sentiment

def combine_features_for_inference(ta_features, sentiment_vector):
    """
    Combine TA features with sentiment features for inference.
    
    Args:
        ta_features: numpy array of shape (context_length, 80) - TA features
        sentiment_vector: numpy array of shape (7,) - News sentiment features
    
    Returns:
        numpy array of shape (context_length, 87) - Combined features
    """
    context_length = ta_features.shape[0]
    
    # Repeat sentiment vector for each time step
    sentiment_features = np.tile(sentiment_vector, (context_length, 1))  # Shape: (context_length, 7)
    
    # Combine TA and sentiment features
    combined_features = np.concatenate([ta_features, sentiment_features], axis=1)  # Shape: (context_length, 87)
    
    return combined_features

def run_inference_example():
    """Example of running inference with the trained model."""
    
    # Model configuration (must match training config)
    model_config = {
        'input_size': 87,  # 80 TA + 7 news sentiment features
        'prediction_length': 10,
        'context_length': 60,
        'd_model': 128,
        'n_head': 8,
        'n_layer': 4,
        'rl_state_size': 256
    }
    
    # Find the most recent trained model
    model_dir = 'tst_model_output'
    if not os.path.exists(model_dir):
        print(f"Model directory {model_dir} not found. Please train the model first.")
        return
    
    model_files = [f for f in os.listdir(model_dir) if f.startswith('tst_model_best_') and f.endswith('.pt')]
    if not model_files:
        print(f"No trained model found in {model_dir}. Please train the model first.")
        return
    
    # Use the most recent model
    model_path = os.path.join(model_dir, sorted(model_files)[-1])
    print(f"Loading model: {model_path}")
    
    # Load the trained model
    model = load_trained_model(model_path, model_config)
    print("Model loaded successfully!")
    
    # Example ticker
    ticker = "AAPL"
    print(f"\n=== Running inference for {ticker} ===")
    
    # 1. Get latest TA features (80 features for 60 days)
    ta_features, dates = get_latest_ta_features(ticker, num_days=60)
    print(f"TA features shape: {ta_features.shape}")
    
    # 2. Get current news sentiment features (7 features)
    sentiment_vector = get_news_sentiment_features(ticker)
    print(f"Sentiment vector shape: {sentiment_vector.shape}")
    
    # 3. Combine features for inference
    combined_features = combine_features_for_inference(ta_features, sentiment_vector)
    print(f"Combined features shape: {combined_features.shape}")
    
    # 4. Prepare input tensor
    input_tensor = torch.FloatTensor(combined_features).unsqueeze(0)  # Add batch dimension: (1, 60, 87)
    print(f"Input tensor shape: {input_tensor.shape}")
    
    # 5. Run inference
    with torch.no_grad():
        # Get RL state for reinforcement learning agent
        rl_state = model(input_tensor)
        print(f"RL state shape: {rl_state.shape}")
        print(f"RL state (first 10 values): {rl_state[0, :10].numpy()}")
        
        # Get future predictions
        predictions = model.predict_future(input_tensor, num_steps=10)
        print(f"Predictions shape: {predictions.shape}")
        print(f"Predicted close price trend (next 10 days): {predictions[0, :, 4].numpy()}")  # Column 4 is 'close'
    
    print("\n=== Inference completed successfully! ===")

if __name__ == "__main__":
    run_inference_example() 
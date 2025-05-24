#!/usr/bin/env python3
"""
Create a small test dataset for testing predict.py
Based on the structure observed in the actual data but much smaller for quick testing.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_test_data():
    """Create a small test dataset with structure matching the training data."""
    
    # Configuration for test data
    num_tickers = 3
    num_days = 100  # Enough for context_length (60) + prediction_length (10) + buffer
    num_features = 80  # TA features (matching train.py expectation)
    
    # Generate tickers
    tickers = ['TEST1', 'TEST2', 'TEST3']
    
    # Generate dates
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]
    
    # Generate feature names (simple TA-like features)
    feature_names = [
        'open', 'high', 'low', 'close', 'volume', 'dividends', 'stock_splits',
        'SMA_10', 'SMA_20', 'SMA_50', 'EMA_10', 'EMA_20', 'EMA_50',
        'MACD', 'RSI_14', 'RSI_21', 'ATR_14', 'OBV', 'CCI_14', 'MFI_14'
    ]
    
    # Add more features to reach 80
    for i in range(len(feature_names), num_features):
        feature_names.append(f'TA_feature_{i}')
    
    print(f"Creating test data with {num_features} features: {feature_names[:10]}...{feature_names[-5:]}")
    
    # Create data for each ticker
    all_data = []
    
    for ticker in tickers:
        print(f"Generating data for {ticker}...")
        
        # Create realistic-looking price data
        base_price = np.random.uniform(50, 200)  # Random starting price
        price_trend = np.random.uniform(-0.001, 0.001)  # Small daily trend
        volatility = np.random.uniform(0.01, 0.03)  # Daily volatility
        
        ticker_data = []
        
        for date in dates:
            # Generate price data with trend and random walk
            if not ticker_data:
                # First day
                open_price = base_price
            else:
                # Subsequent days - use previous close as basis
                open_price = ticker_data[-1]['close'] * (1 + np.random.normal(0, volatility/2))
            
            daily_return = np.random.normal(price_trend, volatility)
            close_price = open_price * (1 + daily_return)
            high_price = max(open_price, close_price) * (1 + abs(np.random.normal(0, volatility/4)))
            low_price = min(open_price, close_price) * (1 - abs(np.random.normal(0, volatility/4)))
            volume = int(np.random.uniform(100000, 10000000))
            
            # Create a row of data
            row = {
                'Date': date.strftime('%Y-%m-%d'),
                'Ticker': ticker,
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'dividends': 0.0,
                'stock_splits': 0.0
            }
            
            # Add technical indicators (simplified calculations)
            # SMA_10, SMA_20, SMA_50
            if len(ticker_data) >= 10:
                sma_10 = np.mean([d['close'] for d in ticker_data[-9:]] + [close_price])
            else:
                sma_10 = close_price
            
            if len(ticker_data) >= 20:
                sma_20 = np.mean([d['close'] for d in ticker_data[-19:]] + [close_price])
            else:
                sma_20 = close_price
                
            if len(ticker_data) >= 50:
                sma_50 = np.mean([d['close'] for d in ticker_data[-49:]] + [close_price])
            else:
                sma_50 = close_price
            
            row.update({
                'SMA_10': sma_10,
                'SMA_20': sma_20,
                'SMA_50': sma_50,
                'EMA_10': sma_10 * (1 + np.random.normal(0, 0.01)),  # Simplified EMA
                'EMA_20': sma_20 * (1 + np.random.normal(0, 0.01)),
                'EMA_50': sma_50 * (1 + np.random.normal(0, 0.01)),
                'MACD': np.random.uniform(-2, 2),
                'RSI_14': np.random.uniform(20, 80),
                'RSI_21': np.random.uniform(20, 80),
                'ATR_14': abs(np.random.normal(1, 0.5)),
                'OBV': volume * np.random.choice([-1, 1]),
                'CCI_14': np.random.uniform(-200, 200),
                'MFI_14': np.random.uniform(0, 100)
            })
            
            # Add remaining features with random values
            for feature in feature_names:
                if feature not in row:
                    if 'volume' in feature.lower():
                        row[feature] = int(np.random.uniform(100000, 1000000))
                    elif any(x in feature.lower() for x in ['rsi', 'mfi', 'stoch']):
                        row[feature] = np.random.uniform(0, 100)
                    elif 'price' in feature.lower() or feature in ['open', 'high', 'low', 'close']:
                        row[feature] = close_price * (1 + np.random.normal(0, 0.01))
                    else:
                        row[feature] = np.random.normal(0, 1)
            
            ticker_data.append(row)
        
        all_data.extend(ticker_data)
    
    # Create DataFrame
    df = pd.DataFrame(all_data)
    
    # Ensure we have the expected number of features
    expected_cols = ['Date', 'Ticker'] + feature_names
    df = df[expected_cols]
    
    print(f"Created test dataset with shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"Tickers: {df['Ticker'].unique()}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    return df

if __name__ == '__main__':
    # Create test data
    test_df = create_test_data()
    
    # Save to CSV
    output_file = 'test_data_small.csv'
    test_df.to_csv(output_file, index=False)
    print(f"\nTest data saved to: {output_file}")
    
    # Show sample
    print(f"\nSample data:")
    print(test_df.head())
    print(f"\nData info:")
    print(test_df.info()) 
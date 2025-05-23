#!/usr/bin/env python3
"""
Test script to validate the data loading and processing logic from train.py
without requiring ML dependencies
"""

import csv
import os
from datetime import datetime

def test_data_loading_logic():
    """Test the data loading and indexing logic from train.py"""
    
    csv_file = "all_tickers_historical_features.csv"
    
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found")
        return
    
    print("Testing data loading logic...")
    
    # Simulate pandas DataFrame loading logic
    data = []
    header = None
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            if len(row) == len(header):
                data.append(row)
    
    print(f"Loaded {len(data)} rows with {len(header)} columns")
    
    # Find key columns
    date_col = header.index('Date') if 'Date' in header else None
    ticker_col = header.index('Ticker') if 'Ticker' in header else None
    
    if date_col is None or ticker_col is None:
        print("Error: Date or Ticker column not found")
        return
    
    print(f"Date column: {date_col}, Ticker column: {ticker_col}")
    
    # Group by ticker and analyze
    ticker_data = {}
    for row in data:
        ticker = row[ticker_col]
        date = row[date_col]
        
        if ticker not in ticker_data:
            ticker_data[ticker] = []
        ticker_data[ticker].append((date, row))
    
    print(f"\nTicker analysis:")
    for ticker in sorted(ticker_data.keys()):
        dates = [item[0] for item in ticker_data[ticker]]
        print(f"  {ticker}: {len(dates)} records, from {min(dates)} to {max(dates)}")
        
        # Check if we have enough data for sequences (60 + 10 = 70 minimum)
        if len(dates) >= 70:
            print(f"    ✓ Sufficient data for training (need 70, have {len(dates)})")
        else:
            print(f"    ✗ Insufficient data for training (need 70, have {len(dates)})")
    
    # Test synthetic news feature addition logic
    print(f"\nTesting synthetic news feature addition:")
    news_features = [
        'avg_sentiment_positive', 'avg_sentiment_negative', 'avg_sentiment_neutral',
        'news_count', 'weekend_effect_positive', 'weekend_effect_negative', 'weekend_effect_neutral'
    ]
    
    # Current features (excluding Date and Ticker which become index)
    current_features = [col for col in header if col not in ['Date', 'Ticker']]
    total_features_after_news = len(current_features) + len(news_features)
    
    print(f"  Current TA features: {len(current_features)}")
    print(f"  News features to add: {len(news_features)}")
    print(f"  Total features after combination: {total_features_after_news}")
    print(f"  Expected model input_size: {total_features_after_news}")
    
    # Test sequence creation logic
    print(f"\nTesting sequence creation logic:")
    context_length = 60
    prediction_length = 10
    
    total_sequences = 0
    for ticker, ticker_rows in ticker_data.items():
        num_rows = len(ticker_rows)
        if num_rows >= context_length + prediction_length:
            sequences_for_ticker = num_rows - context_length - prediction_length + 1
            total_sequences += sequences_for_ticker
            print(f"  {ticker}: {num_rows} rows → {sequences_for_ticker} sequences")
        else:
            print(f"  {ticker}: {num_rows} rows → 0 sequences (insufficient data)")
    
    print(f"\nTotal sequences that would be created: {total_sequences}")
    
    if total_sequences > 0:
        print("✓ Data structure is compatible with training script")
    else:
        print("✗ No sequences can be created - check data requirements")

if __name__ == "__main__":
    test_data_loading_logic() 
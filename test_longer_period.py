#!/usr/bin/env python3
"""Test feature calculation with longer data period"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory for imports
sys.path.append('.')
from feature_engineering.ta_calculator import calculate_technical_indicators

def test_longer_period():
    print('=== 더 긴 기간으로 테스트 (90일) ===')
    
    # Get 90 days of data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    ticker = yf.Ticker('AAPL')
    data = ticker.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))
    data.reset_index(inplace=True)
    
    print(f'다운로드된 데이터: {len(data)}일')
    
    if len(data) >= 50:
        # Calculate features
        features_df = calculate_technical_indicators(data.copy(), include_ohlcv=True)
        
        if not features_df.empty:
            print(f'계산된 총 컬럼: {len(features_df.columns)}개')
            numeric_cols = features_df.select_dtypes(include=['number']).columns
            print(f'숫자형 피처: {len(numeric_cols)}개')
            
            # Check for key missing features
            key_features = ['SMA_50', 'EMA_50', 'MACD_12_26_9', 'UO_7_14_28']
            missing = [f for f in key_features if f not in features_df.columns]
            present = [f for f in key_features if f in features_df.columns]
            
            print(f'핵심 피처 중 생성됨: {present}')
            print(f'핵심 피처 중 누락됨: {missing}')
            
            # Save result
            features_df.to_csv('AAPL_90days_features.csv', index=True)
            print('결과 저장: AAPL_90days_features.csv')
        else:
            print('피처 계산 실패')
    else:
        print('데이터 부족')

if __name__ == '__main__':
    test_longer_period() 
# tst_model/predict_realtime.py: Real-time TST model prediction with live data collection

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
from data_collection.ta_fetcher import fetch_recent_ohlcv_for_inference
from feature_engineering.ta_calculator import calculate_technical_indicators
from data_collection.news_analyzer import fetch_and_analyze_recent_news
from feature_engineering.news_processor import aggregate_daily_sentiment_features

# Configuration matching train.py
DEFAULT_MODEL_CONFIG = {
    'input_size': 87,  # 80 TA + 7 News features
    'prediction_length': 10,
    'context_length': 60,    # Still need 60 days context for TST model
    'n_layer': 4,
    'n_head': 8,
    'd_model': 128,
    'rl_state_size': 256,
    'distribution_output': "normal", 
    'loss': "nll",             
    'num_parallel_samples': 100
}

PREDICT_CONFIG = {
    'model_dir': os.path.join(PROJECT_ROOT, 'tst_model_output'),
    'output_dir': os.path.join(PROJECT_ROOT, 'tst_predictions'),
    'batch_size': 32,
    'business_days_for_analysis': 7,  # Recent 7 business days for analysis
    'lookback_days_for_context': 60   # Additional days needed for TST context
}

def collect_realtime_ta_data(ticker: str, business_days: int = 7, lookback_period: int = 60) -> pd.DataFrame:
    """
    실시간으로 TA 데이터 수집 (ta_fetcher.py 사용)
    
    Args:
        ticker (str): 종목 코드
        business_days (int): 분석 대상 영업일 수
        lookback_period (int): TST 모델 컨텍스트를 위한 추가 일수
        
    Returns:
        pd.DataFrame: TA 특성이 계산된 데이터프레임
    """
    print(f"=== Collecting real-time TA data for {ticker} ===")
    print(f"Target: {business_days} business days + {lookback_period} days context")
    
    # ta_fetcher의 fetch_recent_ohlcv_for_inference 사용
    ohlcv_df = fetch_recent_ohlcv_for_inference(
        ticker_symbol=ticker,
        business_days=business_days,
        lookback_period=lookback_period
    )
    
    if ohlcv_df.empty:
        print(f"❌ No OHLCV data collected for {ticker}")
        return pd.DataFrame()
    
    print(f"✅ OHLCV data collected: {len(ohlcv_df)} rows")
    print(f"Date range: {ohlcv_df['Date'].iloc[0]} to {ohlcv_df['Date'].iloc[-1]}")
    
    # DataFrame을 TA 계산에 맞는 형태로 준비
    try:
        # Date를 인덱스로 설정 (TA 계산에 필요)
        ohlcv_df['Date'] = pd.to_datetime(ohlcv_df['Date'])
        ohlcv_df.set_index('Date', inplace=True)
        
        # 기술적 지표 계산
        print("Calculating technical indicators...")
        ta_df = calculate_technical_indicators(ohlcv_df.copy(), include_ohlcv=True)
        
        if ta_df.empty:
            print(f"❌ Failed to calculate technical indicators for {ticker}")
            return pd.DataFrame()
        
        # Date를 다시 컬럼으로 변환
        ta_df.reset_index(inplace=True)
        ta_df['Date'] = ta_df['Date'].dt.strftime('%Y-%m-%d')
        
        print(f"✅ Technical indicators calculated: {len(ta_df)} rows, {len(ta_df.columns)} features")
        return ta_df
        
    except Exception as e:
        print(f"❌ Error calculating TA for {ticker}: {e}")
        return pd.DataFrame()

def collect_realtime_news_data(ticker: str, analysis_days: int = 7) -> pd.DataFrame:
    """
    실시간으로 뉴스 감성 데이터 수집 (news_analyzer.py + news_processor.py 사용)
    
    Args:
        ticker (str): 종목 코드
        analysis_days (int): 분석 대상 일수
        
    Returns:
        pd.DataFrame: 일별 뉴스 감성 특성 데이터프레임
    """
    print(f"=== Collecting real-time news sentiment for {ticker} ===")
    print(f"Target: {analysis_days} days of news analysis")
    
    try:
        # 뉴스 수집 및 감성 분석
        analyzed_news_df = fetch_and_analyze_recent_news(ticker)
        
        if analyzed_news_df.empty:
            print(f"No news found for {ticker}, creating neutral sentiment")
            return create_neutral_news_sentiment(analysis_days)
        
        print(f"✅ News collected: {len(analyzed_news_df)} articles")
        
        # 일별 감성 특성 집계
        daily_sentiment_df = aggregate_daily_sentiment_features(analyzed_news_df, ticker)
        
        if daily_sentiment_df.empty:
            print(f"Failed to process news sentiment for {ticker}")
            return create_neutral_news_sentiment(analysis_days)
        
        print(f"✅ News sentiment processed: {len(daily_sentiment_df)} days")
        return daily_sentiment_df
        
    except Exception as e:
        print(f"❌ Error collecting news for {ticker}: {e}")
        return create_neutral_news_sentiment(analysis_days)

def create_neutral_news_sentiment(days: int) -> pd.DataFrame:
    """뉴스가 없을 때 중립 감성 데이터 생성"""
    print(f"Creating neutral sentiment data for {days} days")
    
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days-1)
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    neutral_data = {
        'date': date_range,
        'avg_sentiment_positive': 0.0,
        'avg_sentiment_negative': 0.0,
        'avg_sentiment_neutral': 1.0,
        'news_count': 0,
        'weekend_effect_positive': 0.0,
        'weekend_effect_negative': 0.0,
        'weekend_effect_neutral': 0.0,
    }
    
    df = pd.DataFrame(neutral_data)
    df.set_index('date', inplace=True)
    return df

def merge_ta_and_news_data(ta_df: pd.DataFrame, news_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    TA 데이터와 뉴스 감성 데이터를 병합
    
    Args:
        ta_df (pd.DataFrame): TA 특성 데이터
        news_df (pd.DataFrame): 뉴스 감성 데이터
        ticker (str): 종목 코드
        
    Returns:
        pd.DataFrame: 병합된 87개 특성 데이터
    """
    print(f"=== Merging TA and News data for {ticker} ===")
    
    if ta_df.empty:
        print(f"❌ No TA data to merge for {ticker}")
        return pd.DataFrame()
    
    # TA 데이터 준비
    ta_df_copy = ta_df.copy()
    ta_df_copy['Date'] = pd.to_datetime(ta_df_copy['Date'])
    ta_df_copy['date'] = ta_df_copy['Date'].dt.date
    
    # 뉴스 데이터 준비
    if news_df.empty:
        print("No news data, creating neutral sentiment for all TA dates")
        # TA 데이터의 모든 날짜에 대해 중립 감성 생성
        unique_dates = ta_df_copy['date'].unique()
        news_data = {
            'date': unique_dates,
            'avg_sentiment_positive': 0.0,
            'avg_sentiment_negative': 0.0,
            'avg_sentiment_neutral': 1.0,
            'news_count': 0,
            'weekend_effect_positive': 0.0,
            'weekend_effect_negative': 0.0,
            'weekend_effect_neutral': 0.0,
        }
        news_df = pd.DataFrame(news_data)
        news_df.set_index('date', inplace=True)
    
    # 뉴스 데이터 인덱스를 컬럼으로 변환
    news_df_copy = news_df.reset_index()
    
    # 날짜 기준으로 병합
    merged_df = pd.merge(ta_df_copy, news_df_copy, on='date', how='left')
    
    # 뉴스 특성이 없는 날짜는 중립값으로 채움
    news_features = ['avg_sentiment_positive', 'avg_sentiment_negative', 'avg_sentiment_neutral',
                    'news_count', 'weekend_effect_positive', 'weekend_effect_negative', 'weekend_effect_neutral']
    
    for feature in news_features:
        if feature not in merged_df.columns:
            if 'neutral' in feature:
                merged_df[feature] = 1.0
            else:
                merged_df[feature] = 0.0
        else:
            if 'neutral' in feature:
                merged_df[feature].fillna(1.0, inplace=True)
            else:
                merged_df[feature].fillna(0.0, inplace=True)
    
    # Ticker 컬럼 추가
    merged_df['Ticker'] = ticker
    
    # 최종 정리
    merged_df = merged_df.drop(['date'], axis=1)
    merged_df = merged_df.sort_values('Date')
    
    print(f"✅ Data merged successfully: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    
    # 87개 특성 확인
    numeric_cols = merged_df.select_dtypes(include=['number']).columns.tolist()
    print(f"Numeric features: {len(numeric_cols)} (target: 87)")
    
    return merged_df

def load_latest_model(model_dir: str, model_config: dict, device: torch.device):
    """훈련된 TST 모델 로드 (기존 함수와 동일)"""
    model_pattern = os.path.join(model_dir, "tst_model_best_*.pt")
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        raise FileNotFoundError(f"No trained models found in {model_dir}")
    
    latest_model_path = max(model_files, key=os.path.getmtime)
    print(f"Loading model from: {latest_model_path}")
    
    model = TSTModel(config_dict=model_config).to(device)
    checkpoint = torch.load(latest_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, latest_model_path

def prepare_realtime_data_for_prediction(ticker: str, context_length: int = 60):
    """
    실시간 데이터 수집 및 예측용 준비
    
    Args:
        ticker (str): 종목 코드
        context_length (int): TST 모델 컨텍스트 길이
        
    Returns:
        dict: 처리된 데이터 및 메타데이터
    """
    print(f"=== Preparing real-time data for {ticker} ===")
    
    # 1. TA 데이터 수집 (context_length + 분석 대상 일수)
    ta_df = collect_realtime_ta_data(
        ticker=ticker,
        business_days=PREDICT_CONFIG['business_days_for_analysis'],
        lookback_period=context_length
    )
    
    if ta_df.empty:
        raise ValueError(f"Failed to collect TA data for {ticker}")
    
    # 2. 뉴스 데이터 수집
    news_df = collect_realtime_news_data(
        ticker=ticker,
        analysis_days=PREDICT_CONFIG['business_days_for_analysis']
    )
    
    # 3. 데이터 병합
    combined_df = merge_ta_and_news_data(ta_df, news_df, ticker)
    
    if combined_df.empty:
        raise ValueError(f"Failed to merge data for {ticker}")
    
    # 4. DataFrame 구조 설정 (기존 predict.py와 호환)
    combined_df['Date'] = pd.to_datetime(combined_df['Date'])
    combined_df.set_index(['Ticker', 'Date'], inplace=True)
    combined_df.sort_index(inplace=True)
    
    print(f"Final combined data shape: {combined_df.shape}")
    
    # 5. 특성 스케일링
    numeric_cols = combined_df.select_dtypes(include=['number']).columns.tolist()
    print(f"Scaling {len(numeric_cols)} numeric features")
    
    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(combined_df[numeric_cols])
    scaled_df = pd.DataFrame(scaled_values, index=combined_df.index, columns=numeric_cols)
    
    return {
        'scaled_data': scaled_df,
        'scaler': scaler,
        'feature_columns': numeric_cols,
        'raw_data': combined_df
    }

def create_prediction_sequences_realtime(scaled_data: pd.DataFrame, context_length: int):
    """실시간 데이터로 예측 시퀀스 생성 (기존 함수와 유사)"""
    prediction_data = {}
    
    for ticker, group in scaled_data.groupby(level='Ticker'):
        ticker_data = group.values
        
        if len(ticker_data) < context_length:
            print(f"Warning: {ticker} insufficient data ({len(ticker_data)} < {context_length})")
            continue
        
        # 최근 context_length 데이터 포인트 사용
        latest_sequence = ticker_data[-context_length:]
        latest_dates = group.index.get_level_values('Date')[-context_length:]
        
        prediction_data[ticker] = {
            'sequence': torch.FloatTensor(latest_sequence).unsqueeze(0),
            'dates': latest_dates,
            'last_date': latest_dates[-1]
        }
        
        print(f"Prepared sequence for {ticker}: {latest_sequence.shape}")
    
    return prediction_data

def predict_with_tst_model_realtime(model: TSTModel, prediction_data: dict, device: torch.device, mode: str = 'rl_state'):
    """TST 모델로 예측 실행 (기존 함수와 동일)"""
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for ticker, data in prediction_data.items():
            print(f"Predicting for ticker: {ticker}")
            
            sequence = data['sequence'].to(device)
            
            if mode == 'rl_state':
                rl_state = model(past_values=sequence)
                predictions[ticker] = {
                    'rl_state': rl_state.cpu().numpy().squeeze(),
                    'last_date': data['last_date'],
                    'prediction_type': 'rl_state'
                }
                print(f"  RL State shape: {rl_state.shape}")
                
            elif mode == 'forecast':
                forecast = model.predict_future(sequence)
                predictions[ticker] = {
                    'forecast': forecast.cpu().numpy().squeeze(),
                    'last_date': data['last_date'],
                    'prediction_type': 'forecast'
                }
                print(f"  Forecast shape: {forecast.shape}")
                
    return predictions

def save_predictions_realtime(predictions: dict, output_dir: str, model_path: str):
    """예측 결과 저장 (기존 함수와 동일)"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    for ticker, pred_data in predictions.items():
        if pred_data['prediction_type'] == 'rl_state':
            output_file = os.path.join(output_dir, f"{ticker}_rl_state_realtime_{timestamp}.npy")
            np.save(output_file, pred_data['rl_state'])
            print(f"Saved RL state for {ticker}: {output_file}")
            
        elif pred_data['prediction_type'] == 'forecast':
            forecast_df = pd.DataFrame(pred_data['forecast'])
            forecast_df.index.name = 'prediction_day'
            output_file = os.path.join(output_dir, f"{ticker}_forecast_realtime_{timestamp}.csv")
            forecast_df.to_csv(output_file)
            print(f"Saved forecast for {ticker}: {output_file}")
    
    # 요약 파일 저장
    summary_file = os.path.join(output_dir, f"realtime_prediction_summary_{timestamp}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"Real-time TST Model Prediction Summary\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Data Collection: Live (TA + News)\n")
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
    
    print(f"Saved realtime prediction summary: {summary_file}")

def main():
    """실시간 예측 메인 함수"""
    parser = argparse.ArgumentParser(description='Real-time TST Model Prediction')
    parser.add_argument('--ticker', type=str, required=True, help='Ticker to predict (required)')
    parser.add_argument('--mode', type=str, choices=['rl_state', 'forecast'], default='rl_state',
                       help='Prediction mode: rl_state or forecast')
    parser.add_argument('--model_dir', type=str, default=PREDICT_CONFIG['model_dir'],
                       help='Directory containing trained models')
    parser.add_argument('--output_dir', type=str, default=PREDICT_CONFIG['output_dir'],
                       help='Output directory for predictions')
    
    args = parser.parse_args()
    
    print("=== Real-time TST Model Prediction ===")
    print(f"Target ticker: {args.ticker}")
    print(f"Prediction mode: {args.mode}")
    print(f"Model directory: {args.model_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Data source: Real-time (TA + News)")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    try:
        # 1. 실시간 데이터 수집 및 준비
        data_info = prepare_realtime_data_for_prediction(
            ticker=args.ticker,
            context_length=DEFAULT_MODEL_CONFIG['context_length']
        )
        
        # 2. 모델 설정 업데이트
        model_config = DEFAULT_MODEL_CONFIG.copy()
        actual_input_size = len(data_info['feature_columns'])
        model_config['input_size'] = actual_input_size
        print(f"Model input_size: {model_config['input_size']} (from real-time data)")
        
        # 3. 모델 로드
        model, model_path = load_latest_model(args.model_dir, model_config, device)
        
        # 4. 예측 시퀀스 생성
        prediction_data = create_prediction_sequences_realtime(
            data_info['scaled_data'],
            model_config['context_length']
        )
        
        if not prediction_data:
            print("No valid data for prediction. Exiting.")
            return
        
        # 5. 예측 실행
        predictions = predict_with_tst_model_realtime(model, prediction_data, device, mode=args.mode)
        
        # 6. 결과 저장
        save_predictions_realtime(predictions, args.output_dir, model_path)
        
        print("=== Real-time Prediction Complete ===")
        print(f"Processed {len(predictions)} ticker(s)")
        print(f"Results saved to: {args.output_dir}")
        
    except Exception as e:
        print(f"Error during real-time prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main() 
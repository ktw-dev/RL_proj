# main.py: Entry point of the stock trading application. 
"""
[입력 정보]
1. 종목 (ticker)
2. 주식 보유 여부
  2-1. 보유 중인 경우: 보유 수량(num_of_share), 평단가(avg_price)
3. 현재 사용 가능한 투자 금액 (bullet)
4. 사용자가 원하는 액션 (매수: 1, 매도: 0) 및 해당 의사의 강도(가중치)
5. 선택할 에이전트: PPO(‘bob’) 또는 SAC(‘sara’)

[내부 처리 과정]
1. 입력 정보를 기반으로 ticker, num_of_share, bullet, action_intention 등 핵심 상태 변수 정리
2. 최근 7일 간의 뉴스 수집 → 감성 분석 수행
   - news_fetcher.py → news_processor.py 파이프라인
   - 출력: 일별 감성 점수 리스트
3. 기술적 분석 데이터 수집 (OHLCV + 지표)
   - ta_fetcher.py 통해 7일치 OHLCV 및 기술 지표 생성
4. TST 기반 가격 예측 수행
   - 입력: OHLCV + 기술 지표 + 감성 점수
   - 출력: 향후 가격 변화 예측값
5. 상태(state) 벡터 구성
   - 구성 요소: 예측값 + 보유 정보 + 감성 점수 + 사용 가능 자금 등
6. 선택된 에이전트(PPO/SAC)를 통해 최적 액션 도출
   - action = agent.predict(state)
7. 최종 출력
   - 종목의 향후 예측 방향 (상승/하락)
   - 추천 액션 (매수/매도/관망)
   - 권장 가격대 (예: X원 이하에 매수, Y원 이상에 매도 등)

[비고]
- 전체 로직은 main.py에서 통합적으로 실행되며, 사용자 입력부터 최종 트레이딩 가이드 출력까지 자동화됨.
"""

import sys
from pathlib import Path
import pandas as pd # For type hinting and potential DataFrame manipulation
from datetime import datetime, timedelta, timezone
import os
import glob

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# --- Actual Module Imports ---
from data_collection.news_analyzer import fetch_and_analyze_recent_news
from feature_engineering.news_processor import aggregate_daily_sentiment_features, is_us_business_day
from data_collection.ta_fetcher import fetch_recent_ohlcv_for_inference
from feature_engineering.ta_calculator_now import calculate_current_technical_indicators
from feature_engineering.ta_calculator import calculate_technical_indicators
from tst_model.model import TSTModel
from config.tickers import SUPPORTED_TICKERS
import torch
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# --- TST Model Configuration (matching train.py and predict_realtime.py) ---
DEFAULT_MODEL_CONFIG = {
    'input_size': 87,  # 80 TA + 7 News features
    'prediction_length': 10,
    'context_length': 60,    # TST model context length
    'n_layer': 4,
    'n_head': 8,
    'd_model': 128,
    'rl_state_size': 256,
    'distribution_output': "normal", 
    'loss': "nll",             
    'num_parallel_samples': 100
}

# --- Constants for Data Fetching/Processing ---
TA_LOOKBACK_PERIOD = 60  # Days of historical data needed for TA calculation (e.g., for 50-day SMA)
TST_MODEL_DIR = os.path.join(project_root, 'tst_model_output')

# --- Mock Agent Classes (to be replaced with real implementations later) ---
class MockPPOAgent:
    def predict_action(self, state_vector: dict):
        print(f"[Mock PPO] Analyzing state: price={state_vector.get('current_price')}, sentiment={state_vector.get('market_sentiment_score'):.3f}")
        # Simple mock logic based on sentiment and price trend
        sentiment = state_vector.get('market_sentiment_score', 0.0)
        tst_direction = state_vector.get('tst_predicted_direction', 'UNKNOWN')
        current_price = state_vector.get('current_price', 0)
        
        if tst_direction == "UP" and sentiment > 0.1:
            action = "BUY"
            reason = "Positive TST prediction and sentiment"
            target_price = current_price * 1.02 if current_price else None
        elif tst_direction == "DOWN" or sentiment < -0.1:
            action = "SELL"
            reason = "Negative TST prediction or sentiment"
            target_price = current_price * 0.98 if current_price else None
        else:
            action = "HOLD"
            reason = "Uncertain market conditions"
            target_price = current_price
            
        return {"action": action, "reason": reason, "target_price": target_price}

class MockSACAgent:
    def predict_action(self, state_vector: dict):
        print(f"[Mock SAC] Analyzing state: price={state_vector.get('current_price')}, sentiment={state_vector.get('market_sentiment_score'):.3f}")
        # Mock SAC with slightly different logic
        sentiment = state_vector.get('market_sentiment_score', 0.0)
        confidence = state_vector.get('tst_confidence', 0.0)
        current_price = state_vector.get('current_price', 0)
        
        if confidence > 0.7 and sentiment > 0.05:
            action = "BUY"
            reason = "High confidence TST prediction with positive sentiment"
            target_price = current_price * 1.015 if current_price else None
        elif confidence > 0.7 and sentiment < -0.05:
            action = "SELL" 
            reason = "High confidence TST prediction with negative sentiment"
            target_price = current_price * 0.985 if current_price else None
        else:
            action = "HOLD"
            reason = "Low confidence or neutral sentiment"
            target_price = current_price
            
        return {"action": action, "reason": reason, "target_price": target_price}

# --- Helper Function Definitions --- 

def get_user_input():
    """Collects necessary input from the user through CLI."""
    print("\n--- 1. Collecting User Input ---")

    # 지원하는 종목
    print(f"지원하는 종목: {SUPPORTED_TICKERS}")
    
    # 1. 종목 입력
    ticker = input(f"종목을 입력하세요 ({', '.join(SUPPORTED_TICKERS)} 중 선택): ").strip().upper()
    if ticker not in SUPPORTED_TICKERS:
        raise ValueError(f"Ticker {ticker} is not supported.")

    # 2. 주식 보유 여부
    has_shares_input = input("해당 종목을 보유하고 있습니까? (y/n): ").strip().lower()
    has_shares = has_shares_input == "y"

    num_shares = 0
    avg_price = 0.0
    if has_shares:
        num_shares = int(input("보유 주식 수를 입력하세요: "))
        avg_price = float(input("보유 주식의 평단가를 입력하세요: "))

    # 3. 현재 사용 가능한 투자금
    bullet = float(input("현재 사용 가능한 투자 금액을 입력하세요: "))

    # 4. 사용자 의향 (매수/매도)
    action_str = input("원하는 액션을 입력하세요 (BUY/SELL): ").strip().upper()
    strength = float(input("이 액션을 얼마나 원하십니까? (0.0 ~ 1.0): "))
    if action_str not in ["BUY", "SELL"]:
        raise ValueError("Action must be BUY or SELL.")

    user_action_intention = {"action": action_str, "strength": strength}

    # 5. 에이전트 선택
    agent_choice = input("사용할 에이전트를 선택하세요 (PPO/SAC): ").strip().upper()
    if agent_choice not in ["PPO", "SAC"]:
        raise ValueError("Agent must be PPO or SAC.")

    user_input = {
        "ticker": ticker,
        "has_shares": has_shares,
        "num_shares": num_shares,
        "avg_price": avg_price,
        "bullet": bullet,
        "user_action_intention": user_action_intention,
        "agent_choice": agent_choice
    }

    print(f"\n[입력 확인] {user_input}")
    return user_input

def initialize_state_variables(user_input: dict):
    """Processes raw user input into structured state variables.
    Returns a dictionary of core state variables.
    """
    print("\n--- 2. Initializing State Variables ---")
    state_vars = {
        "ticker": user_input["ticker"],
        "holdings_info": {
            "has_shares": user_input["has_shares"],
            "num_shares": user_input.get("num_shares", 0),
            "avg_price": user_input.get("avg_price", 0.0)
        },
        "available_cash": user_input["bullet"],
        "user_intention": user_input["user_action_intention"],
        "chosen_agent": user_input["agent_choice"]
    }
    print(f"Initialized State: {state_vars}")
    return state_vars

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
    TA 데이터와 뉴스 감성 데이터를 병합하여 87개 특성 데이터 생성
    
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
    
    # 날짜 타입 통일 (object 타입으로 변환)
    ta_df_copy['date'] = ta_df_copy['date'].astype(str)
    if 'date' in news_df_copy.columns:
        news_df_copy['date'] = pd.to_datetime(news_df_copy['date']).dt.date.astype(str)
    
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

def load_latest_tst_model(model_dir: str, model_config: dict, device: torch.device):
    """훈련된 TST 모델 로드"""
    model_pattern = os.path.join(model_dir, "tst_model_best_*.pt")
    model_files = glob.glob(model_pattern)
    
    if not model_files:
        raise FileNotFoundError(f"No trained models found in {model_dir}")
    
    latest_model_path = max(model_files, key=os.path.getmtime)
    print(f"Loading TST model from: {latest_model_path}")
    
    model = TSTModel(config_dict=model_config).to(device)
    checkpoint = torch.load(latest_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"TST Model loaded successfully. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, latest_model_path

def predict_price_with_tst_model(technical_data: pd.DataFrame, daily_sentiment: pd.Series, ticker: str):
    """Predicts future price movement using the TST model.
    Input: TA data and daily sentiment scores.
    Returns: A dictionary with price prediction details including RL state.
    """
    print("\n--- 5. Predicting Price with TST Model ---")
    
    if technical_data.empty:
        print("Skipping TST prediction due to missing technical data.")
        return {"predicted_direction": "UNKNOWN", "confidence": 0.0, "rl_state": None}

    try:
        # Setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 1. TA 데이터와 뉴스 데이터 병합하여 87개 특성 생성
        # technical_data를 DataFrame으로 변환 (인덱스가 date이므로)
        ta_df = technical_data.reset_index()
        ta_df['Date'] = pd.to_datetime(ta_df['Date'])
        
        # 뉴스 감성을 DataFrame으로 변환
        if not daily_sentiment.empty:
            news_df = pd.DataFrame({
                'date': daily_sentiment.index,
                'avg_sentiment_positive': np.where(daily_sentiment > 0, daily_sentiment, 0),
                'avg_sentiment_negative': np.where(daily_sentiment < 0, -daily_sentiment, 0),
                'avg_sentiment_neutral': np.where(daily_sentiment == 0, 1.0, 0.0),
                'news_count': 1,
                'weekend_effect_positive': 0.0,
                'weekend_effect_negative': 0.0,
                'weekend_effect_neutral': 0.0,
            })
            news_df.set_index('date', inplace=True)
        else:
            news_df = pd.DataFrame()
        
        # 데이터 병합
        combined_df = merge_ta_and_news_data(ta_df, news_df, ticker)
        
        if combined_df.empty:
            print("Failed to merge TA and news data")
            return {"predicted_direction": "UNKNOWN", "confidence": 0.0, "rl_state": None}
        
        # 2. 모델 설정 및 로드
        model_config = DEFAULT_MODEL_CONFIG.copy()
        numeric_cols = combined_df.select_dtypes(include=['number']).columns.tolist()
        model_config['input_size'] = len(numeric_cols)
        
        print(f"Loading TST model with input_size: {model_config['input_size']}")
        model, model_path = load_latest_tst_model(TST_MODEL_DIR, model_config, device)
        
        # 3. 데이터 전처리
        combined_df['Date'] = pd.to_datetime(combined_df['Date'])
        combined_df.set_index(['Ticker', 'Date'], inplace=True)
        combined_df.sort_index(inplace=True)
        
        # 스케일링
        scaler = MinMaxScaler()
        scaled_values = scaler.fit_transform(combined_df[numeric_cols])
        scaled_df = pd.DataFrame(scaled_values, index=combined_df.index, columns=numeric_cols)
        
        # 4. 예측 시퀀스 생성 (최근 60일 사용)
        ticker_data = scaled_df.values
        context_length = model_config['context_length']
        
        if len(ticker_data) < context_length:
            print(f"Insufficient data for TST prediction ({len(ticker_data)} < {context_length})")
            return {"predicted_direction": "UNKNOWN", "confidence": 0.0, "rl_state": None}
        
        # 최근 context_length 데이터 사용
        sequence = ticker_data[-context_length:]
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0).to(device)
        
        # 5. RL State 예측
        with torch.no_grad():
            rl_state = model(past_values=sequence_tensor)
            rl_state_numpy = rl_state.cpu().numpy().squeeze()
            
            # 예측 결과 분석 (간단한 휴리스틱)
            rl_state_mean = np.mean(rl_state_numpy)
            rl_state_std = np.std(rl_state_numpy)
            
            # 방향 예측 (RL state의 평균값 기반)
            if rl_state_mean > 0.1:
                predicted_direction = "UP"
                confidence = min(0.9, abs(rl_state_mean) * 2)
            elif rl_state_mean < -0.1:
                predicted_direction = "DOWN" 
                confidence = min(0.9, abs(rl_state_mean) * 2)
            else:
                predicted_direction = "SIDEWAYS"
                confidence = 0.3
            
            print(f"TST RL State: shape={rl_state_numpy.shape}, mean={rl_state_mean:.4f}, std={rl_state_std:.4f}")
            print(f"Predicted direction: {predicted_direction}, confidence: {confidence:.2f}")
            
            return {
                "predicted_direction": predicted_direction,
                "confidence": confidence,
                "rl_state": rl_state_numpy,
                "rl_state_mean": rl_state_mean,
                "rl_state_std": rl_state_std
            }
        
    except Exception as e:
        print(f"Error in TST prediction: {e}")
        import traceback
        traceback.print_exc()
        return {"predicted_direction": "UNKNOWN", "confidence": 0.0, "rl_state": None}

def construct_rl_state_vector(current_state_vars: dict, technical_data: pd.DataFrame, 
                                daily_sentiment: pd.Series, tst_prediction: dict):
    """Constructs the state vector for the RL agent.
    Combines various pieces of information into a format suitable for the agent.
    Returns: A dictionary representing the state.
    """
    print("\n--- 6. Constructing State Vector for RL Agent ---")
    
    # 기본 정보 추출 (컬럼명 체크)
    if not technical_data.empty:
        # 'Close' 또는 'close' 컬럼 찾기
        if 'Close' in technical_data.columns:
            current_price = technical_data['Close'].iloc[-1]
        elif 'close' in technical_data.columns:
            current_price = technical_data['close'].iloc[-1]
        else:
            print(f"Warning: No Close/close column found. Available columns: {technical_data.columns.tolist()}")
            current_price = None
    else:
        current_price = None
        
    latest_sentiment = daily_sentiment.iloc[-1] if not daily_sentiment.empty else 0.0

    # TST RL State 사용 (실제 256차원 벡터)
    tst_rl_state = tst_prediction.get("rl_state")
    
    state_vector = {
        "ticker": current_state_vars["ticker"],
        "current_price": current_price,
        "market_sentiment_score": latest_sentiment,
        "tst_predicted_direction": tst_prediction.get("predicted_direction"),
        "tst_confidence": tst_prediction.get("confidence"),
        "tst_rl_state": tst_rl_state,  # 256차원 RL state 벡터
        "tst_rl_state_mean": tst_prediction.get("rl_state_mean"),
        "tst_rl_state_std": tst_prediction.get("rl_state_std"),
        "has_shares": current_state_vars["holdings_info"]["has_shares"],
        "num_shares": current_state_vars["holdings_info"]["num_shares"],
        "avg_purchase_price": current_state_vars["holdings_info"]["avg_price"],
        "available_cash": current_state_vars["available_cash"],
        "user_action_preference": current_state_vars["user_intention"]
    }
    
    print(f"RL State Vector constructed:")
    print(f"  Current price: {current_price}")
    print(f"  Sentiment: {latest_sentiment:.3f}")
    print(f"  TST RL State: {type(tst_rl_state)} shape={tst_rl_state.shape if tst_rl_state is not None else None}")
    print(f"  TST Direction: {tst_prediction.get('predicted_direction')}")
    print(f"  TST Confidence: {tst_prediction.get('confidence'):.3f}")
    
    return state_vector

def get_action_from_rl_agent(state_vector: dict, agent_choice: str):
    """Gets the trading action from the chosen RL agent.
    Uses: ppo_agent.py or sac_agent.py (mocked here)
    Returns: A dictionary with the agent's action and reasoning.
    """
    print(f"\n--- 7. Getting Action from RL Agent ({agent_choice}) ---")
    agent = None
    if agent_choice == "PPO":
        agent = MockPPOAgent()
    elif agent_choice == "SAC":
        agent = MockSACAgent()
    else:
        print(f"Warning: Unknown agent choice '{agent_choice}'. Defaulting to PPO.")
        agent = MockPPOAgent() 
        # Or raise an error: raise ValueError("Invalid agent choice")

    if not state_vector.get("current_price"):
         print("Skipping RL action due to missing current price in state vector.")
         return {"action": "HOLD", "reason": "Insufficient data for RL decision"}

    rl_action_decision = agent.predict_action(state_vector)
    print(f"RL Agent Decision: {rl_action_decision}")
    return rl_action_decision

def generate_final_recommendation(tst_prediction: dict, rl_decision: dict, ticker: str):
    """Generates the final trading recommendation for the user.
    Combines TST insights and RL agent action into human-readable advice.
    Returns: A string with the recommendation.
    """
    print("\n--- 8. Generating Final Trading Recommendation ---")
    recommendation = f"--- Trading Recommendation for {ticker} ---\n"
    recommendation += f"Market Outlook (TST): {tst_prediction.get('predicted_direction', 'N/A')} (Confidence: {tst_prediction.get('confidence', 0)*100:.0f}%)\n"
    if tst_prediction.get('rl_state') is not None:
        recommendation += f"TST RL State Analysis: Mean={tst_prediction.get('rl_state_mean', 0):.4f}, Std={tst_prediction.get('rl_state_std', 0):.4f}\n"
    
    recommendation += f"RL Agent Action: {rl_decision.get('action', 'N/A')}\n"
    recommendation += f"Reason: {rl_decision.get('reason', 'N/A')}\n"
    if rl_decision.get('target_price') is not None:
        recommendation += f"Suggested Price Level: {rl_decision['target_price']:.2f}\n"
    
    print(recommendation)
    return recommendation

# --- Main Execution Flow --- 
def run_trading_bot():
    """Main function to run the trading bot logic.
    """
    try:
        # 1. Get user input
        user_input = get_user_input()
        
        # 2. Initialize core state variables from user input
        current_state = initialize_state_variables(user_input)
        ticker = current_state["ticker"]
        days_to_analyze = 7 # As per specification for news and TA

        # 3. Collect real-time TA data
        current_utc_time = datetime.now(timezone.utc)
        print(f"\n=== Starting Real-time Data Collection for {ticker} ===")
        
        ta_df = collect_realtime_ta_data(
            ticker=ticker,
            business_days=days_to_analyze,
            lookback_period=TA_LOOKBACK_PERIOD
        )
        
        if ta_df.empty:
            print("❌ Failed to collect TA data. Aborting.")
            return
        
        # 4. Collect real-time news sentiment data
        news_df = collect_realtime_news_data(
            ticker=ticker,
            analysis_days=days_to_analyze
        )
        
        # 뉴스 데이터를 기존 형식에 맞게 변환
        if not news_df.empty:
            # combined_score 계산
            news_df['combined_score'] = (
                news_df['avg_sentiment_positive'] - news_df['avg_sentiment_negative'] +
                news_df['weekend_effect_positive'] - news_df['weekend_effect_negative']
            )
            daily_sentiment_scores = news_df['combined_score']
        else:
            # 빈 시리즈 생성
            daily_sentiment_scores = pd.Series(dtype=float, name='combined_score')
        
        # TA 데이터를 기존 형식에 맞게 변환
        ta_df['Date'] = pd.to_datetime(ta_df['Date'])
        ta_df.set_index('Date', inplace=True)
        
        # 최근 days_to_analyze일 필터링
        relevant_dates = []
        current_d = current_utc_time.date() - timedelta(days=1)
        while len(relevant_dates) < days_to_analyze:
            if is_us_business_day(current_d):
                relevant_dates.append(current_d)
            current_d -= timedelta(days=1)
        relevant_dates.sort()
        
        technical_data = ta_df.reindex(pd.to_datetime(relevant_dates))
        technical_data.index = pd.to_datetime(technical_data.index).date
        technical_data.index.name = 'Date'
        technical_data.fillna(0.0, inplace=True)
        
        print(f"✅ Final data prepared:")
        print(f"  TA data: {technical_data.shape}")
        print(f"  News sentiment: {len(daily_sentiment_scores)} entries")
        
        # 5. Predict price movement using TST model
        tst_price_prediction = predict_price_with_tst_model(technical_data, daily_sentiment_scores, ticker)
        
        # 6. Construct the state vector for the RL agent
        rl_state = construct_rl_state_vector(current_state, technical_data, daily_sentiment_scores, tst_price_prediction)
        
        # 7. Get action from the chosen RL agent
        rl_action = get_action_from_rl_agent(rl_state, current_state["chosen_agent"])
        
        # 8. Generate final trading recommendation
        final_recommendation_message = generate_final_recommendation(tst_price_prediction, rl_action, ticker)
        
        print("\n=== Trading Bot Cycle Completed Successfully ===")
        print(f"✅ Real-time data collection: TA + News")
        print(f"✅ TST Model prediction: RL State generated")
        print(f"✅ RL Agent decision: {rl_action.get('action')}")
        print(f"✅ Final recommendation provided")

    except ValueError as ve:
        print(f"Input Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # In a real application, add more detailed error logging here

if __name__ == "__main__":
    run_trading_bot() 
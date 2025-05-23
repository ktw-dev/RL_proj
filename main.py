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
   - news_fetcher.py → sentiment_analyzer.py → news_processor.py 파이프라인
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

# Add the project root directory to sys.path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# --- Actual Module Imports ---
from data_collection.news_analyzer import fetch_and_analyze_recent_news
from feature_engineering.news_processor import aggregate_daily_sentiment_features, is_us_business_day
from data_collection.ta_fetcher import fetch_recent_ohlcv_for_inference
from feature_engineering.ta_calculator_now import calculate_current_technical_indicators
# from models import tst_predictor # Assuming a module for TST model - Kept for other mocks
# from agents import ppo_agent, sac_agent # Assuming modules for RL agents - Kept for other mocks
from config.tickers import SUPPORTED_TICKERS # Assuming this exists - Kept if used by other mocks or main logic

# For demonstration, let's define placeholder functions/classes for these imports
# In a real scenario, these would be in separate files as commented above.

# MockNewsFetcher, MockSentimentAnalyzer, MockNewsProcessor are removed as they are replaced by real imports for news processing.

# --- Constants for Data Fetching/Processing ---
TA_LOOKBACK_PERIOD = 60  # Days of historical data needed for TA calculation (e.g., for 50-day SMA)

class MockTSTPredictor:
    def predict_price_change(self, technical_data: pd.DataFrame, daily_sentiment_scores: pd.Series):
        print(f"[Mock] Predicting price change using TST with TA data (shape: {technical_data.shape}) and sentiment (len: {len(daily_sentiment_scores)}).")
        # Returns a prediction (e.g., expected price change or direction)
        return {"predicted_direction": "UP", "confidence": 0.75, "next_day_price_change_percentage": 0.01}

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

def collect_and_process_news_sentiment(ticker: str, analysis_date_utc: datetime, days_for_ta_window: int = 7):
    """특정 종목에 대한 뉴스 감성을 수집, 분석 및 처리합니다."""
    print(f"\n--- 3. {ticker}에 대한 뉴스 감성 수집, 분석 및 처리 중 ---")
    
    # 1단계: 뉴스 수집 및 헤드라인별 감성 분석
    analyzed_headlines_df = fetch_and_analyze_recent_news(
        ticker_symbol=ticker,
        analysis_base_date_utc=analysis_date_utc 
    )

    if analyzed_headlines_df.empty:
        print(f"{ticker}에 대한 헤드라인을 찾거나 분석할 수 없습니다. 일별 감성을 처리할 수 없습니다.")
        return pd.Series(dtype=float) 

    # 2단계: 일별 감성 집계 및 주말 로직 적용
    daily_sentiment_features_df = aggregate_daily_sentiment_features(
        analyzed_news_df=analyzed_headlines_df,
        ticker_symbol=ticker
    )

    if daily_sentiment_features_df.empty:
        print(f"{ticker}에 대한 일별 감성 특성 처리에 실패했습니다.")
        return pd.Series(dtype=float)
    
    # TST 모델을 위한 단일 '결합 감성 점수' 계산
    daily_sentiment_features_df['combined_score'] = (
        daily_sentiment_features_df['avg_sentiment_positive'] - daily_sentiment_features_df['avg_sentiment_negative'] +
        daily_sentiment_features_df['weekend_effect_positive'] - daily_sentiment_features_df['weekend_effect_negative']
    )
    
    # analysis_date_utc 기준 관련 날짜 필터링 (TA 윈도우와 일치하도록)
    relevant_dates = []
    current_d = analysis_date_utc.date() - timedelta(days=1)
    while len(relevant_dates) < days_for_ta_window:
        if is_us_business_day(current_d):
            relevant_dates.append(current_d)
        current_d -= timedelta(days=1)
    relevant_dates.sort() # 시간순 정렬

    final_sentiment_series = daily_sentiment_features_df['combined_score'].reindex(relevant_dates).fillna(0.0)
    
    print(f"TST 모델용 일별 감성 시리즈 처리 (샘플):\n{final_sentiment_series.tail()}")
    return final_sentiment_series


def collect_technical_analysis_data(ticker: str, days: int = 7, analysis_date_utc: datetime = None):
    """Fetches OHLCV and technical indicators for the specified period.
    Uses: data_collection.ta_fetcher and feature_engineering.ta_calculator_now
    Returns: pd.DataFrame with OHLCV and TA data, indexed by Date (datetime.date objects).
    """
    print(f"\n--- 4. Collecting Technical Analysis Data for {ticker} (target: last {days} business days) ---")

    if analysis_date_utc is None:
        analysis_date_utc = datetime.now(timezone.utc)

    ohlcv_data = fetch_recent_ohlcv_for_inference(
        ticker_symbol=ticker,
        business_days=days,
        lookback_period=TA_LOOKBACK_PERIOD
    )

    if ohlcv_data.empty:
        print(f"Failed to fetch OHLCV data for {ticker}. Skipping TA calculation.")
        return pd.DataFrame()

    features_df = calculate_current_technical_indicators(ohlcv_data, include_ohlcv=True)

    if features_df.empty:
        print(f"Failed to calculate technical indicators for {ticker}.")
        return pd.DataFrame()

    relevant_dates = []
    current_d = analysis_date_utc.date() - timedelta(days=1)
    while len(relevant_dates) < days:
        if is_us_business_day(current_d):
            relevant_dates.append(current_d)
        current_d -= timedelta(days=1)
    relevant_dates.sort()

    try:
        if not isinstance(features_df.index, pd.DatetimeIndex):
            print("Warning: features_df index is not DatetimeIndex. Attempting conversion.")
            features_df.index = pd.to_datetime(features_df.index)
        
        # Reindex with Timestamps, then convert final index to date objects
        aligned_technical_data = features_df.reindex(pd.to_datetime(relevant_dates))
        
        if not aligned_technical_data.empty:
            aligned_technical_data.index = pd.to_datetime(aligned_technical_data.index).date
            aligned_technical_data.index.name = 'Date'
            aligned_technical_data.fillna(0.0, inplace=True) # Fill NaNs from reindexing or earlier steps
            print(f"Technical Analysis Data prepared for {ticker} (shape: {aligned_technical_data.shape}):\\n{aligned_technical_data.tail()}")
        else:
            print(f"Technical data became empty after reindexing for {ticker}. This might indicate date mismatches.")
            return pd.DataFrame()
            
        return aligned_technical_data

    except Exception as e:
        print(f"Error aligning technical data for {ticker}: {e}")
        return pd.DataFrame()

def predict_price_with_tst_model(technical_data: pd.DataFrame, daily_sentiment: pd.Series):
    """Predicts future price movement using the TST model.
    Uses: tst_predictor.py (mocked here)
    Input: TA data and daily sentiment scores.
    Returns: A dictionary with price prediction details.
    """
    print("\n--- 5. Predicting Price with TST Model ---")
    if technical_data.empty:
        print("Skipping TST prediction due to missing technical data.")
        return {"predicted_direction": "UNKNOWN", "confidence": 0.0}

    predictor = MockTSTPredictor()
    # In a real scenario, data alignment (dates) between TA and sentiment would be crucial here.
    # For mock, we assume they are implicitly aligned or predictor handles it.
    price_prediction = predictor.predict_price_change(technical_data, daily_sentiment)
    print(f"TST Price Prediction: {price_prediction}")
    return price_prediction

def construct_rl_state_vector(current_state_vars: dict, technical_data: pd.DataFrame, 
                                daily_sentiment: pd.Series, tst_prediction: dict):
    """Constructs the state vector for the RL agent.
    Combines various pieces of information into a format suitable for the agent.
    Returns: A dictionary representing the state.
    """
    print("\n--- 6. Constructing State Vector for RL Agent ---")
    # This is highly dependent on the RL agent's expected input.
    # Example components:
    current_price = technical_data['Close'].iloc[-1] if not technical_data.empty else None
    latest_sentiment = daily_sentiment.iloc[-1] if not daily_sentiment.empty else 0.0

    state_vector = {
        "ticker": current_state_vars["ticker"],
        "current_price": current_price,
        "market_sentiment_score": latest_sentiment, # Example: latest daily sentiment
        "tst_predicted_direction": tst_prediction.get("predicted_direction"),
        "tst_confidence": tst_prediction.get("confidence"),
        "has_shares": current_state_vars["holdings_info"]["has_shares"],
        "num_shares": current_state_vars["holdings_info"]["num_shares"],
        "avg_purchase_price": current_state_vars["holdings_info"]["avg_price"],
        "available_cash": current_state_vars["available_cash"],
        "user_action_preference": current_state_vars["user_intention"] # User's desired action & strength
        # Potentially more features from technical_data (e.g., latest indicators)
        # or historical sentiment might be included, flattened as needed.
    }
    print(f"RL State Vector (sample): {{current_price: {current_price}, sentiment: {latest_sentiment}, ...}}")
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
    recommendation = f"--- Trading Recommendation for {ticker} ---\
"
    recommendation += f"Market Outlook (TST): {tst_prediction.get('predicted_direction', 'N/A')} (Confidence: {tst_prediction.get('confidence', 'N/A')*100:.0f}%)\n"
    if 'next_day_price_change_percentage' in tst_prediction:
        recommendation += f"Expected Price Change (next day): {tst_prediction['next_day_price_change_percentage']*100:.2f}%\n"
    
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

        # 3. Collect news data and perform sentiment analysis
        # The 'days' parameter in the original call was ambiguous.
        # Assuming 'days_to_analyze' refers to 'days_for_ta_window' and analysis is based on current UTC time.
        current_utc_time = datetime.now(timezone.utc)
        daily_sentiment_scores = collect_and_process_news_sentiment(
            ticker, 
            analysis_date_utc=current_utc_time, 
            days_for_ta_window=days_to_analyze
        )
        
        # 4. Collect technical analysis data (OHLCV + Indicators)
        technical_data = collect_technical_analysis_data(
            ticker, 
            days=days_to_analyze,
            analysis_date_utc=current_utc_time # Pass the same analysis date
        )
        
        # 5. Predict price movement using TST model
        tst_price_prediction = predict_price_with_tst_model(technical_data, daily_sentiment_scores)
        
        # 6. Construct the state vector for the RL agent
        rl_state = construct_rl_state_vector(current_state, technical_data, daily_sentiment_scores, tst_price_prediction)
        
        # 7. Get action from the chosen RL agent
        rl_action = get_action_from_rl_agent(rl_state, current_state["chosen_agent"])
        
        # 8. Generate final trading recommendation
        final_recommendation_message = generate_final_recommendation(tst_price_prediction, rl_action, ticker)
        
        # (Optional) Further actions: e.g., logging, order execution interface call
        print("\nTrading bot cycle finished.")

    except ValueError as ve:
        print(f"Input Error: {ve}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        # In a real application, add more detailed error logging here

if __name__ == "__main__":
    run_trading_bot() 
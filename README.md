# Stock Trading Bot with TST and RL 

count_column.py 실행 결과
```
기본: 82개
마지막 5개: ['open_Z_30_1', 'high_Z_30_1', 'low_Z_30_1', 'close_Z_30_1', 'Ticker']
BOM 제거: 82개
숫자형: 80개
전체 컬럼:
 1: 'Date'
 2: 'open'
 3: 'high'
 4: 'low'
 5: 'close'
 6: 'volume'
 7: 'dividends'
 8: 'stock splits'
 9: 'SMA_10'
10: 'SMA_20'
11: 'SMA_50'
12: 'EMA_10'
13: 'EMA_20'
14: 'EMA_50'
15: 'WMA_10'
16: 'WMA_20'
17: 'DEMA_10'
18: 'DEMA_20'
19: 'TEMA_10'
20: 'TEMA_20'
21: 'TRIMA_10'
22: 'TRIMA_20'
23: 'KAMA_10'
24: 'KAMA_20'
25: 'T3_10'
26: 'VWAP_D'
27: 'MACD_12_26_9'
28: 'MACDh_12_26_9'
29: 'MACDs_12_26_9'
30: 'MOM_10'
31: 'MOM_20'
32: 'RSI_14'
33: 'RSI_21'
34: 'STOCHk_14_3_3'
35: 'STOCHd_14_3_3'
36: 'STOCHFk_5_3_1'
37: 'STOCHFd_5_3_1'
38: 'STOCHRSIk_14_14_3_3'
39: 'STOCHRSId_14_14_3_3'
40: 'WILLR_14'
41: 'CCI_14'
42: 'CCI_20'
43: 'CMO_14'
44: 'ROC_10'
45: 'ROC_20'
46: 'ROCR_10'
47: 'APO_12_26'
48: 'PPO_12_26_9'
49: 'PPOh_12_26_9'
50: 'PPOs_12_26_9'
51: 'UO_7_14_28'
52: 'ADX_14'
53: 'DMP_14'
54: 'DMN_14'
55: 'AROOND_14'
56: 'AROONU_14'
57: 'AROONOSC_14'
58: 'PDM_14'
59: 'MDM_14'
60: 'BBL_20_2'
61: 'BBM_20_2'
62: 'BBU_20_2'
63: 'BBB_20_2'
64: 'BBP_20_2'
65: 'ATR_14'
66: 'NATR_14'
67: 'TRUERANGE'
68: 'MIDPOINT_14'
69: 'MIDPRICE_14'
70: 'KCUe_20_10'
71: 'KCLe_20_10'
72: 'KCM_20_10'
73: 'OBV'
74: 'AD'
75: 'ADOSC_3_10'
76: 'MFI_14'
77: 'BOP'
78: 'open_Z_30_1'
79: 'high_Z_30_1'
80: 'low_Z_30_1'
81: 'close_Z_30_1'
82: 'Ticker'
```

```
Using found data file: /home/theta/Public/reinforcement_project-main/all_tickers_historical_features.csv
Starting TST model training...
Using device: cuda
Loading historical TA data from /home/theta/Public/reinforcement_project-main/all_tickers_historical_features.csv
Loaded TA data. Shape: (130923, 80)
Tickers found: ['AAPL', 'AMD', 'AMZN', 'ASML', 'AVGO', 'AZN', 'COST', 'CSCO', 'GOOG', 'GOOGL', 'INTU', 'ISRG', 'LIN', 'META', 'MSFT', 'NFLX', 'NVDA', 'PLTR', 'TMUS', 'TSLA']
Found 80 numeric TA features: ['open', 'high', 'low', 'close', 'volume']...['BOP', 'open_Z_30_1', 'high_Z_30_1', 'low_Z_30_1', 'close_Z_30_1']
Adding synthetic neutral news sentiment features for training...
Total features after adding news sentiment: 87 (80 TA + 7 news)
Updated model_config input_size to: 87
Final features shape: (130923, 87)
Scaling features per ticker...
Scaled AAPL: 10943 samples
Scaled AMD: 11131 samples
Scaled AMZN: 6791 samples
Scaled ASML: 7340 samples
Scaled AVGO: 3715 samples
Scaled AZN: 7805 samples
Scaled COST: 9536 samples
Scaled CSCO: 8622 samples
Scaled GOOG: 4965 samples
Scaled GOOGL: 4965 samples
Scaled INTU: 7847 samples
Scaled ISRG: 6012 samples
Scaled LIN: 8033 samples
Scaled META: 3013 samples
Scaled MSFT: 9617 samples
Scaled NFLX: 5529 samples
Scaled NVDA: 6366 samples
Scaled PLTR: 908 samples
Scaled TMUS: 4295 samples
Scaled TSLA: 3490 samples
Features scaled. Final shape: (130923, 87)
Creating sequences for TST model...
Created sequences from 20 tickers
Created 129543 sequences.
Past sequences shape: torch.Size([129543, 60, 87]), Future sequences shape: torch.Size([129543, 10, 87])
Train loader: 3239 batches, Val loader: 810 batches
Initializing TST model...
Model initialized with 1,848,279 parameters
Starting training loop...
Early stopping patience: 15
Minimum improvement delta: 1e-06
Epoch 1/100, Batch 20/3239, Train Loss: 0.164203, LR: 3.09e-07
Epoch 1/100, Batch 40/3239, Train Loss: 0.158850, LR: 6.17e-07
Epoch 1/100, Batch 60/3239, Train Loss: 0.180978, LR: 9.26e-07
Epoch 1/100, Batch 80/3239, Train Loss: 0.190384, LR: 1.23e-06
Epoch 1/100, Batch 100/3239, Train Loss: 0.188278, LR: 1.54e-06
Epoch 1/100, Batch 120/3239, Train Loss: 0.161388, LR: 1.85e-06
Epoch 1/100, Batch 140/3239, Train Loss: 0.175591, LR: 2.16e-06
Epoch 1/100, Batch 160/3239, Train Loss: 0.176010, LR: 2.47e-06
...
```

---

## `main.py`: Application Entry Point and Orchestrator

`main.py`는 전체 주식 트레이딩 봇 애플리케이션의 진입점이자 핵심 로직을 통합적으로 관리하는 오케스트레이터 역할을 수행합니다. 사용자 입력부터 최종 트레이딩 조언 출력까지의 전 과정을 자동화된 파이프라인으로 구성합니다.

### 주요 기능 및 처리 흐름

`main.py`의 `run_trading_bot()` 함수를 중심으로 다음과 같은 단계로 작동합니다:

1.  **사용자 입력 수집 (`get_user_input`)**
    *   **목적**: 트레이딩 결정에 필요한 초기 정보를 사용자로부터 받습니다.
    *   **입력**: (애플리케이션 실행 시 CLI 또는 GUI를 통해)
        *   거래 대상 종목 (`ticker`)
        *   현재 주식 보유 상태 (보유 여부, 보유량, 평단가)
        *   사용 가능한 투자금 (`bullet`)
        *   사용자의 거래 의도 (선호 액션: 매수/매도, 의사 강도)
        *   선호하는 RL 에이전트 (`PPO` 또는 `SAC`)
    *   **출력**: 사용자의 입력 정보를 담은 딕셔너리.

2.  **핵심 상태 변수 초기화 (`initialize_state_variables`)**
    *   **목적**: 수집된 사용자 입력을 분석 및 거래 로직에 사용될 내부 상태 변수로 가공합니다.
    *   **입력**: `get_user_input`으로부터 받은 사용자 정보 딕셔너리.
    *   **출력**: 구조화된 현재 상태 정보를 담은 딕셔너리 (예: `current_state`).

3.  **뉴스 데이터 수집 및 감성 분석 (`collect_and_process_news_sentiment`)**
    *   **목적**: 선택된 종목에 대한 최근 뉴스(기본 7일)를 수집하고, 감성 분석을 통해 시장의 심리를 파악합니다.
    *   **가정 의존성**: 
        *   `data_collection.news_fetcher`: 특정 종목의 뉴스를 가져오는 모듈.
        *   `analysis.sentiment_analyzer`: 뉴스 텍스트의 감성을 분석하는 모듈.
        *   `analysis.news_processor`: 분석된 감성 점수를 일별로 집계하는 모듈.
    *   **입력**: 종목 코드 (`ticker`), 분석 기간 (일수).
    *   **출력**: 일별 평균 감성 점수를 담은 Pandas Series (날짜 인덱스).

4.  **기술적 분석 데이터 수집 (`collect_technical_analysis_data`)**
    *   **목적**: 주가 예측 및 RL 에이전트의 상태 구성에 필요한 OHLCV 및 각종 기술적 지표를 수집합니다 (기본 7일).
    *   **가정 의존성**: 
        *   `data_collection.ta_fetcher`: OHLCV 및 기술적 지표를 계산하여 제공하는 모듈.
    *   **입력**: 종목 코드 (`ticker`), 분석 기간 (일수).
    *   **출력**: OHLCV 및 기술적 지표를 포함하는 Pandas DataFrame (날짜 인덱스).

5.  **TST 모델 기반 가격 예측 (`predict_price_with_tst_model`)**
    *   **목적**: 수집된 기술적 데이터와 뉴스 감성 점수를 통합하여 Time Series Transformer(TST) 모델을 통해 향후 주가 움직임을 예측합니다.
    *   **가정 의존성**: 
        *   `models.tst_predictor`: 학습된 TST 모델을 로드하고 예측을 수행하는 모듈.
    *   **입력**: 기술적 분석 데이터 (DataFrame), 일별 감성 점수 (Series).
    *   **출력**: 가격 예측 결과 딕셔너리 (예: 예상 방향, 신뢰도, 예상 변동폭 등).

6.  **RL 에이전트 상태 벡터 구성 (`construct_rl_state_vector`)**
    *   **목적**: 강화학습(RL) 에이전트가 최적의 행동을 결정하는 데 필요한 모든 정보를 집약하여 상태(state) 벡터를 생성합니다.
    *   **입력**: 초기 상태 변수, 기술적 데이터, 일별 감성 점수, TST 가격 예측 결과.
    *   **출력**: RL 에이전트의 입력으로 사용될 상태 벡터 (딕셔너리 형태).

7.  **RL 에이전트를 통한 액션 결정 (`get_action_from_rl_agent`)**
    *   **목적**: 구성된 상태 벡터를 바탕으로 사용자가 선택한 RL 에이전트(PPO 또는 SAC)를 사용하여 최적의 거래 액션(매수, 매도, 관망)을 도출합니다.
    *   **가정 의존성**: 
        *   `agents.ppo_agent`: PPO 알고리즘 기반 RL 에이전트.
        *   `agents.sac_agent`: SAC 알고리즘 기반 RL 에이전트.
    *   **입력**: 상태 벡터, 사용자가 선택한 에이전트 이름.
    *   **출력**: RL 에이전트가 제안하는 액션 및 관련 정보 (예: 이유, 목표 가격 등)를 담은 딕셔너리.

8.  **최종 트레이딩 조언 생성 (`generate_final_recommendation`)**
    *   **목적**: TST 모델의 시장 예측과 RL 에이전트의 행동 결정을 종합하여 사용자에게 이해하기 쉬운 최종 트레이딩 조언을 생성합니다.
    *   **입력**: TST 가격 예측 결과, RL 에이전트의 결정.
    *   **출력**: 사용자에게 제시될 최종 추천 메시지 (문자열).

### 실행 방법

프로젝트 루트 디렉토리에서 다음 명령어를 통해 `main.py`를 실행할 수 있습니다:

```bash
python main.py
```

애플리케이션 실행 시, 콘솔을 통해 필요한 정보(모의 입력 방식 사용 시에는 하드코딩된 값 사용)를 입력받아 전체 분석 및 조언 생성 과정을 수행합니다.

### 모듈 의존성

*   `config/`: 전역 설정 (예: API 키, 지원 티커 리스트)
*   `data_collection/`: 데이터 수집 관련 모듈 (`news_fetcher.py`, `ta_fetcher.py`)
*   `analysis/`: 데이터 분석 관련 모듈 (`sentiment_analyzer.py`, `news_processor.py`)
*   `models/`: 예측 모델 관련 모듈 (`tst_predictor.py`)
*   `agents/`: 강화학습 에이전트 관련 모듈 (`ppo_agent.py`, `sac_agent.py`)

---
## Data Collection Scripts

### ta_fetcher.py

`ta_fetcher.py`는 단일 주식 종목의 최근 60일간의 OHLCV(Open, High, Low, Close, Volume) 데이터와 기술적 지표(Technical Indicators)를 수집하고 계산하는 스크립트입니다.

#### 입력 (Inputs)
- **종목 코드 (Ticker Symbol)**
  - `config/tickers.py`에 정의된 `SUPPORTED_TICKERS` 리스트 내의 종목만 처리 가능
  - 스크립트 내의 `target_ticker` 변수를 수정하여 원하는 종목 선택 (기본값: "AAPL")
- **날짜 범위**
  - 자동으로 계산됨:
    - 종료일(end_date): 스크립트 실행 시점
    - 시작일(start_date): 종료일로부터 30일 전

#### 처리 과정 (Process)
1. **데이터 수집**
   - yfinance API를 통해 지정된 종목의 OHLCV 데이터 수집
   - 설정된 날짜 범위(30일) 동안의 일별 데이터 획득

2. **데이터 전처리**
   - 'Date' 컬럼을 datetime 형식으로 변환
   - DatetimeIndex로 설정하여 시계열 데이터 구조 생성

3. **기술적 지표 계산**
   - `feature_engineering/ta_calculator.py`의 `calculate_technical_indicators` 함수 사용
   - OHLCV 데이터를 기반으로 다양한 기술적 지표 계산
   - 원본 OHLCV 데이터도 포함 (include_ohlcv=True)

#### 출력 (Outputs)
- **CSV 파일**
  - 파일명: `{종목코드}_last_30days_features.csv` (예: `AAPL_last_30days_features.csv`)
  - 포함 데이터:
    - 인덱스: Date (날짜)
    - 컬럼: 
      - OHLCV (Open, High, Low, Close, Volume)
      - 계산된 모든 기술적 지표들

#### 실행 방법
```bash
# 프로젝트 루트 디렉토리에서 실행
python data_collection/ta_fetcher.py
```

#### 주의사항
- 스크립트는 반드시 프로젝트 루트 디렉토리에서 실행해야 합니다
- 스크립트는 `config/tickers.py`에 정의된 종목만 처리 가능
- 다른 종목을 처리하려면 스크립트 내의 `target_ticker` 값을 수정해야 함
- `feature_engineering/ta_calculator.py` 모듈이 필요하며 Python 경로에서 접근 가능해야 함
- 실행 시점으로부터 30일 전의 데이터만 처리됨

---

## News Processor (`news_processor.py`)

### 개요

`feature_engineering/news_processor.py`는 뉴스 감성 분석 결과를 일별 특성으로 집계하는 핵심 모듈입니다. 개별 헤드라인의 감성 점수를 일별로 통합하고, 주말 뉴스의 영향을 다음 주 영업일에 전파하는 고급 기능을 제공합니다.

### 주요 기능

#### 1. **일별 감성 집계**
- 개별 뉴스 헤드라인의 감성 점수를 일별로 평균화
- 하루 내 여러 뉴스가 있을 경우 감성 점수 통계 처리
- 뉴스 개수 카운팅으로 시장 관심도 측정

#### 2. **주말 효과 모델링**
- 주말 발행 뉴스의 감성이 다음 주 영업일에 미치는 영향 정량화
- 토요일/일요일 뉴스 → 다음 주 월~금요일 5영업일에 효과 전파
- 복수 주말 뉴스 시 평균 효과 계산

#### 3. **영업일 기반 처리**
- 미국 증시 영업일 (월~금) 기준으로 데이터 정렬
- 주말/공휴일 제외 처리
- 영업일 시퀀스 자동 생성

### 핵심 함수 분석

#### `is_us_business_day(dt_date: date) -> bool`
```python
def is_us_business_day(dt_date: date):
    # Monday(0) to Friday(4)
    return dt_date.weekday() < 5
```
- **목적**: 해당 날짜가 미국 영업일인지 판단
- **기준**: 월요일(0) ~ 금요일(4)
- **제한**: 연방 공휴일은 고려하지 않음 (단순화)

#### `get_next_n_business_days(start_date: date, n: int) -> List[date]`
```python
def get_next_n_business_days(start_date: date, n: int):
    business_days = []
    current_date = start_date
    while len(business_days) < n:
        if is_us_business_day(current_date):
            business_days.append(current_date)
        current_date += timedelta(days=1)
    return business_days
```
- **목적**: 시작일부터 N개의 영업일 생성
- **사용**: 주말 뉴스 효과를 전파할 영업일 계산

#### `aggregate_daily_sentiment_features(analyzed_news_df: pd.DataFrame, ticker_symbol: str) -> pd.DataFrame`

주요 집계 함수로 다음과 같은 단계로 처리됩니다:

### 데이터 처리 파이프라인

#### 1. **입력 데이터 검증**
```python
required_cols = ['published_date', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
```
- **필수 컬럼**: 발행일, 3가지 감성 점수
- **날짜 형식 통일**: Timestamp → date 객체로 변환

#### 2. **일별 기본 집계**
```python
daily_aggregated_sentiments = df.groupby('published_date').agg(
    avg_sentiment_positive=('sentiment_positive', 'mean'),
    avg_sentiment_negative=('sentiment_negative', 'mean'),
    avg_sentiment_neutral=('sentiment_neutral', 'mean'),
    news_count=('headline', 'count')
)
```

#### 3. **주말 효과 전파**
```python
# 주말 뉴스 식별
if not is_us_business_day(current_date): # 토요일 또는 일요일
    # 다음 월요일부터 5영업일에 효과 전파
    start_propagation_date = current_date + timedelta(days=(7 - current_date.weekday()))
    target_business_days = get_next_n_business_days(start_propagation_date, 5)
```

#### 4. **데이터 병합 및 최종 처리**
```python
# 평일 집계 + 주말 효과를 합성
final_df = pd.merge(all_dates.to_frame(), daily_aggregated_sentiments, on='date', how='left')
final_df = pd.merge(final_df, weekend_effects_df, on='date', how='left')
```

### 출력 특성

#### 생성되는 컬럼들

| 컬럼명 | 설명 | 범위 |
|--------|------|------|
| `avg_sentiment_positive` | 일별 평균 긍정 감성 | 0.0 ~ 1.0 |
| `avg_sentiment_negative` | 일별 평균 부정 감성 | 0.0 ~ 1.0 |
| `avg_sentiment_neutral` | 일별 평균 중립 감성 | 0.0 ~ 1.0 |
| `news_count` | 일별 뉴스 개수 | 정수 |
| `weekend_effect_positive` | 주말 뉴스 긍정 효과 | 0.0 ~ 1.0 |
| `weekend_effect_negative` | 주말 뉴스 부정 효과 | 0.0 ~ 1.0 |
| `weekend_effect_neutral` | 주말 뉴스 중립 효과 | 0.0 ~ 1.0 |

#### 데이터 구조
- **인덱스**: `date` (날짜별 정렬)
- **대상**: 영업일 중심 (주말은 효과로만 반영)
- **결측값 처리**: 뉴스가 없는 날은 0.0으로 채움

### 주말 효과 로직 상세

#### 전파 메커니즘
```
토요일 뉴스 → 다음 주 월~금 (5영업일)
일요일 뉴스 → 다음 주 월~금 (5영업일)
```

#### 예시: 2023년 10월 21일(토) 뉴스
- **발행일**: 2023-10-21 (토요일)
- **영향 대상**: 2023-10-23(월) ~ 2023-10-27(금)
- **효과**: 토요일 뉴스의 감성 점수가 5영업일에 동일하게 적용

#### 복수 주말 뉴스 처리
```python
# 같은 영업일에 여러 주말 뉴스 효과가 겹치면 평균 계산
weekend_effects_df = weekend_effects_df.groupby('date').agg({
    'weekend_effect_positive': 'mean',
    'weekend_effect_negative': 'mean',
    'weekend_effect_neutral': 'mean'
})
```

### 실제 사용 예시

#### 입력 데이터 (분석된 뉴스)
```python
analyzed_news_df = pd.DataFrame([
    {'published_date': date(2023,10,20), 'sentiment_positive': 0.3, 'sentiment_negative': 0.1, 'sentiment_neutral': 0.6},  # 금요일
    {'published_date': date(2023,10,21), 'sentiment_positive': 0.8, 'sentiment_negative': 0.1, 'sentiment_neutral': 0.1},  # 토요일 (주말)
    {'published_date': date(2023,10,23), 'sentiment_positive': 0.2, 'sentiment_negative': 0.3, 'sentiment_neutral': 0.5},  # 월요일
])
```

#### 출력 결과
```python
# 2023-10-20 (금요일)
avg_sentiment_positive: 0.3, weekend_effect_positive: 0.0

# 2023-10-23 (월요일) 
avg_sentiment_positive: 0.2, weekend_effect_positive: 0.8  # 토요일 뉴스 효과
```

### 특징 및 장점

#### 1. **현실적인 시장 모델링**
- 주말 뉴스가 월요일 시장에 미치는 영향을 정량화
- 영업일 기준 데이터 정렬로 실제 거래일정과 일치

#### 2. **Robust한 데이터 처리**
- 결측값 자동 처리 (뉴스 없는 날 = 0)
- 날짜 형식 자동 변환
- 에러 상황 대응 (빈 데이터, 컬럼 누락)

#### 3. **유연한 집계 방식**
- 하루 내 복수 뉴스의 평균 감성 계산
- 주말 효과의 평균화 처리
- 확장 가능한 감성 특성 구조

#### 4. **통계적 근거**
- 감성 점수의 확률적 분포 (positive + negative + neutral = 1.0)
- 뉴스 빈도를 통한 시장 관심도 측정
- 시간 지연 효과 모델링

### 통합 역할

이 모듈은 주식 트레이딩 봇 시스템에서 다음과 같은 역할을 수행합니다:

1. **감성 신호 생성**: 원시 뉴스 텍스트 → 정량적 감성 지표
2. **시간 정렬**: 뉴스 시점과 거래 시점 간의 시간 정합성 확보  
3. **특성 엔지니어링**: ML 모델이 사용할 수 있는 형태로 데이터 변환
4. **시장 심리 반영**: 투자자들의 감정적 반응을 수치화

이를 통해 TST 모델과 RL 에이전트가 **뉴스 기반 시장 센티멘트**를 의사결정에 활용할 수 있게 됩니다.

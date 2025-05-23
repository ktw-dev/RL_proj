# Stock Trading Bot with TST and RL 

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

### 모듈 의존성 (가정)

`main.py`는 다음과 같은 주요 모듈(디렉토리)에 의존하는 것으로 가정하고 설계되었습니다. 실제 구현 시 각 모듈 내의 해당 기능들이 개발되어야 합니다.

*   `config/`: 전역 설정 (예: API 키, 지원 티커 리스트)
*   `data_collection/`: 데이터 수집 관련 모듈 (`news_fetcher.py`, `ta_fetcher.py`)
*   `analysis/`: 데이터 분석 관련 모듈 (`sentiment_analyzer.py`, `news_processor.py`)
*   `models/`: 예측 모델 관련 모듈 (`tst_predictor.py`)
*   `agents/`: 강화학습 에이전트 관련 모듈 (`ppo_agent.py`, `sac_agent.py`)

---
## Data Collection Scripts

### Using main.py

프로그램의 진입점인 `main.py`를 통해 데이터 수집을 실행할 수 있습니다.

#### 프로그래매틱 사용
```python
from main import fetch_recent_data, fetch_all_tickers

# 단일 종목 데이터 수집
success = fetch_recent_data("AAPL", days=30)

# 모든 지원 종목 데이터 수집
results = fetch_all_tickers(days=30)
```

#### 커맨드 라인 실행
```bash
# 프로젝트 루트 디렉토리에서 실행
python main.py
```

### ta_fetcher.py

`ta_fetcher.py`는 단일 주식 종목의 최근 30일간의 OHLCV(Open, High, Low, Close, Volume) 데이터와 기술적 지표(Technical Indicators)를 수집하고 계산하는 스크립트입니다.

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

## TST Model (Time Series Transformer) Architecture

### 개요

`tst_model/model.py`에 정의된 `TSTModel`은 시계열 예측과 강화학습을 연결하는 핵심 모델입니다. Hugging Face의 `TimeSeriesTransformerForPrediction`을 기반으로 하여 주식 시계열 데이터를 분석하고, 강화학습 에이전트가 사용할 수 있는 상태 벡터를 생성합니다.

### 모델 구조

#### 기본 아키텍처
```
입력 시계열 데이터 (past_values)
       ↓
TimeSeriesTransformerForPrediction
       ↓
시계열 예측 결과 (distribution parameters)
       ↓
RL Head (Linear Layer)
       ↓
RL State Vector
```

#### 주요 구성 요소

1. **TimeSeriesTransformerForPrediction**
   - Hugging Face의 사전 훈련된 시계열 트랜스포머 모델
   - Encoder-Decoder 구조로 과거 시계열 데이터를 기반으로 미래 예측 수행
   - 확률적 예측 결과 제공 (평균, 분산 등)

2. **RL Head (Linear Layer)**
   - 트랜스포머의 예측 결과를 RL 상태 벡터로 변환
   - 입력 차원: `prediction_length × input_size`
   - 출력 차원: `rl_state_size` (사용자 정의)

### 핵심 매개변수

| 매개변수 | 설명 | 기본값 |
|---------|------|--------|
| `input_size` | 입력 특성 수 (기술적 지표 + 감성 점수) | 필수 |
| `prediction_length` | 예측 시간 범위 (몇 일 후까지 예측) | 필수 |
| `context_length` | 히스토리 시간 범위 (몇 일 전 데이터 사용) | prediction_length |
| `rl_state_size` | RL 상태 벡터 차원 | 필수 |
| `n_layer` | Encoder/Decoder 레이어 수 | 2 |
| `n_head` | Attention Head 수 | 2 |
| `d_model` | 트랜스포머 모델 차원 | 32 |

### 모델 처리 과정

#### 1. 초기화 단계 (`__init__`)
```python
# 설정 매개변수 구성
transformer_config_params = {
    'input_size': self.input_size,
    'prediction_length': self.prediction_length,
    'context_length': self.context_length,
    'encoder_layers': config_dict.get('n_layer', 2),
    'd_model': config_dict.get('d_model', 32),
    # ... 기타 설정
}

# TimeSeriesTransformerConfig 생성
transformer_config = TimeSeriesTransformerConfig(**transformer_config_params)

# 모델 구성요소 생성
self.transformer = TimeSeriesTransformerForPrediction(config=transformer_config)
self.rl_head = nn.Linear(self.rl_head_input_dim, self.rl_state_size)
```

#### 2. Forward Pass (`forward`)

**Training 모드**: 
- 입력: `past_values` (과거 시계열), `future_values` (정답 데이터)
- 출력: `TimeSeriesTransformerOutput` (loss 포함)
- 목적: 모델 학습을 위한 손실 계산

**Inference 모드**:
- 입력: `past_values` (과거 시계열만)
- 처리:
  1. 트랜스포머가 미래 시계열 분포 예측
  2. 예측 분포에서 평균값 추출
  3. 평균값을 Flatten하여 1차원으로 변환
  4. RL Head를 통해 상태 벡터 생성
- 출력: RL 상태 벡터

### 예측 결과 추출 방법

모델은 Hugging Face 트랜스포머의 다양한 출력 형태에 robust하게 대응합니다:

1. **`transformer_output.params[0]`**: 예측 분포의 평균 (가장 일반적)
2. **`transformer_output.prediction_outputs`**: 직접적인 예측 결과
3. **`transformer_output.sequences`**: generate() 메서드 결과
4. **Fallback**: generate() 메서드 재시도 또는 더미 텐서

### 데이터 형태

#### 입력 데이터
- **`past_values`**: `(batch_size, context_length, input_size)`
  - 과거 시계열 데이터 (기술적 지표 + 감성 점수)
- **`future_values`** (훈련시만): `(batch_size, prediction_length, input_size)`
  - 정답 미래 시계열 데이터

#### 출력 데이터
- **훈련시**: `TimeSeriesTransformerOutput` (loss, logits 등 포함)
- **추론시**: `(batch_size, rl_state_size)` 형태의 RL 상태 벡터

### 주요 특징

1. **이중 목적 설계**: 시계열 예측 학습과 RL 상태 생성을 동시에 지원
2. **유연한 출력 처리**: 다양한 Hugging Face 모델 출력 형태에 적응
3. **자동 차원 조정**: 예측 결과와 RL Head 입력 차원 불일치 시 자동 패딩/잘라내기
4. **확률적 예측**: 단순 값이 아닌 분포 기반 예측으로 불확실성 고려

### 사용 예시

```python
# 모델 설정
model_config = {
    'input_size': 50,          # TA 지표 + 감성 점수
    'prediction_length': 10,   # 10일 후 예측
    'context_length': 60,      # 60일 히스토리 사용
    'rl_state_size': 256,      # RL 상태 벡터 크기
    'n_layer': 4,              # 트랜스포머 레이어
    'd_model': 128             # 모델 차원
}

# 모델 생성
tst_model = TSTModel(model_config)

# 훈련
tst_model.train()
output = tst_model(past_values=past_data, future_values=future_data)
loss = output.loss

# RL 상태 벡터 생성
tst_model.eval()
with torch.no_grad():
    rl_state = tst_model(past_values=past_data)  # (batch_size, 256)
```

### 통합적 역할

TST 모델은 주식 트레이딩 봇에서 다음과 같은 핵심 역할을 수행합니다:

1. **시장 동향 예측**: 기술적 지표와 뉴스 감성을 종합하여 미래 주가 방향 예측
2. **불확실성 정량화**: 확률적 예측을 통해 시장의 불확실성 측정
3. **RL 브릿지**: 시계열 예측 결과를 RL 에이전트가 이해할 수 있는 형태로 변환
4. **멀티모달 융합**: 다양한 데이터 소스(가격, 기술적 지표, 감성)를 통합 처리

---

## TST Model Training Script (`train.py`)

### 개요

`tst_model/train.py`는 TST 모델을 사전 훈련(pre-training)하기 위한 스크립트입니다. 기술적 분석 데이터와 뉴스 감성 데이터를 결합하여 시계열 트랜스포머 모델을 학습시키며, 다양한 고급 학습 기법들을 포함하고 있습니다.

### 훈련 데이터

#### 주요 데이터 소스

1. **기술적 분석(TA) 데이터**
   - **파일**: `all_tickers_historical_features.csv`
   - **내용**: 81개 기술적 지표 (OHLCV + 각종 TA 지표)
   - **구조**: MultiIndex (Date, Ticker)로 구성된 시계열 데이터
   - **기간**: 여러 종목의 히스토리컬 데이터

2. **뉴스 감성 데이터 (합성)**
   - **생성 방식**: 실제 뉴스 데이터가 없을 때 중립적 감성 데이터를 합성
   - **특성**: 7개 뉴스 관련 특성
     - `avg_sentiment_positive`, `avg_sentiment_negative`, `avg_sentiment_neutral`
     - `news_count`
     - `weekend_effect_positive`, `weekend_effect_negative`, `weekend_effect_neutral`
   - **기본값**: 모든 감성은 중립(neutral=1.0, positive=negative=0.0)

3. **결합 데이터**
   - **최종 특성 수**: 88개 (81 TA + 7 News)
   - **결합 방식**: `feature_engineering.feature_combiner.align_and_combine_features` 사용
   - **인덱스 정렬**: Date-Ticker 기준으로 정렬된 시계열 데이터

### 훈련 설정 및 파라미터

#### 모델 설정 (`DEFAULT_MODEL_CONFIG`)

| 파라미터 | 값 | 설명 |
|---------|----|----|
| `input_size` | 88 | 입력 특성 수 (81 TA + 7 News) |
| `prediction_length` | 10 | 미래 예측 범위 (10일) |
| `context_length` | 60 | 히스토리 범위 (60일) |
| `n_layer` | 3 | Encoder/Decoder 레이어 수 |
| `n_head` | 4 | Attention Head 수 |
| `d_model` | 128 | 트랜스포머 모델 차원 |
| `rl_state_size` | 256 | RL 상태 벡터 크기 |
| `distribution_output` | "normal" | 예측 분포 (가우시안) |
| `loss` | "nll" | 손실 함수 (음의 로그 우도) |
| `num_parallel_samples` | 100 | 샘플링 시 병렬 샘플 수 |

#### 훈련 설정 (`TRAIN_CONFIG`)

| 파라미터 | 값 | 설명 |
|---------|----|----|
| `data_path` | `all_tickers_historical_features.csv` | 훈련 데이터 경로 |
| `output_dir` | `tst_model_output/` | 모델 저장 디렉토리 |
| `batch_size` | 32 | 배치 크기 |
| `epochs` | 50 | 최대 에포크 수 |
| `learning_rate` | 1e-4 | 학습률 |
| `weight_decay` | 0.01 | 가중치 감쇠 |
| `patience_early_stopping` | 5 | 조기 종료 인내심 |
| `validation_split_ratio` | 0.2 | 검증 데이터 비율 |
| `random_seed` | 42 | 랜덤 시드 |

### 훈련 과정

#### 1. 데이터 준비 단계
```
TA 데이터 로드 → 중립 뉴스 데이터 생성 → 특성 결합 → 스케일링 → 시퀀스 생성
```

**데이터 전처리**:
- **MultiIndex 설정**: (Date, Ticker) 기준으로 인덱싱
- **특성 스케일링**: 종목별 MinMaxScaler 적용 (0-1 정규화)
- **시퀀스 생성**: 슬라이딩 윈도우 방식으로 (과거, 미래) 시퀀스 쌍 생성

#### 2. 시퀀스 생성 로직
```python
# 각 종목별로 시퀀스 생성
for ticker in tickers:
    for i in range(len(data) - context_length - prediction_length + 1):
        past_seq = data[i : i + context_length]          # 60일 과거 데이터
        future_seq = data[i + context_length : i + context_length + prediction_length]  # 10일 미래 데이터
```

#### 3. 모델 훈련 단계
```
모델 초기화 → 옵티마이저 설정 → 학습률 스케줄러 → 훈련 루프 → 검증 → 조기 종료
```

**핵심 구성요소**:
- **옵티마이저**: AdamW (weight decay 포함)
- **스케줄러**: Linear warmup scheduler
- **손실 함수**: NLL (Negative Log-Likelihood)
- **평가 지표**: 검증 손실

### 부가적인 기능

#### 1. **자동 더미 데이터 생성**
```python
# 훈련 데이터가 없을 때 자동으로 더미 데이터 생성
if not os.path.exists(data_file_path):
    num_tickers = 2
    num_days_per_ticker = context_length + prediction_length + 20
    # 랜덤 TA 특성 81개로 더미 데이터 생성
```

#### 2. **종목별 데이터 스케일링**
- 각 종목별로 독립적인 MinMaxScaler 적용
- 종목 간 스케일 차이로 인한 편향 방지
- 스케일러 객체 저장으로 역변환 가능

#### 3. **Robust 시퀀스 생성**
```python
def create_sequences(data_df, context_length, prediction_length, target_cols_indices=None):
    # 데이터 충분성 검사
    if len(ticker_data) < context_length + prediction_length:
        print(f"Skipping ticker {ticker} due to insufficient data")
        continue
```

#### 4. **Early Stopping 메커니즘**
- 검증 손실이 개선되지 않으면 훈련 자동 중단
- 과적합 방지 및 최적 모델 자동 저장
- `patience` 파라미터로 인내심 조절

#### 5. **동적 입력 크기 조정**
```python
# 실제 결합된 특성 수에 따라 input_size 자동 업데이트
model_config['input_size'] = combined_features_df.shape[1]
```

#### 6. **실시간 훈련 모니터링**
- 배치별 손실 로깅 (10 배치마다)
- 에포크별 훈련/검증 손실 요약
- GPU/CPU 자동 감지 및 활용

#### 7. **타임스탬프 기반 모델 저장**
```python
model_path = f"tst_model_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pt"
```

### 훈련 결과물

#### 저장되는 파일들
1. **모델 체크포인트**: `tst_model_best_YYYYMMDD_HHMMSS.pt`
2. **스케일러 객체**: 각 종목별 MinMaxScaler (메모리에 저장)
3. **훈련 로그**: 콘솔 출력으로 손실 기록

#### 모델 성능 지표
- **최종 검증 손실**: 최적 모델의 검증 손실값
- **수렴 에포크**: 조기 종료가 발생한 에포크
- **시퀀스 수**: 생성된 총 훈련 시퀀스 개수

### 실행 방법

```bash
# 프로젝트 루트 디렉토리에서 실행
python tst_model/train.py
```

#### 필수 조건
- `all_tickers_historical_features.csv` 파일 존재 (없으면 자동 더미 데이터 생성)
- `feature_engineering.feature_combiner` 모듈 접근 가능
- 충분한 메모리 (시퀀스 데이터 로딩용)
- GPU 환경 권장 (자동 감지)

### 특징 및 장점

1. **자동화된 파이프라인**: 데이터 로딩부터 모델 저장까지 완전 자동화
2. **Robust한 에러 처리**: 데이터 부족, 파일 누락 등 다양한 상황 대응
3. **확장 가능한 구조**: 새로운 특성이나 종목 추가 시 자동 적응
4. **재현 가능한 실험**: 랜덤 시드 고정으로 일관된 결과
5. **효율적인 메모리 사용**: 종목별 처리로 메모리 효율성 확보

이 훈련 스크립트는 TST 모델을 실제 금융 데이터로 학습시키기 위한 **산업급 파이프라인**을 제공하며, 연구 환경과 프로덕션 환경 모두에서 활용 가능하도록 설계되었습니다.

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

---

## TST Model Prediction Script (`predict.py`)

### 개요

`tst_model/predict.py`는 훈련된 TST 모델을 사용하여 실제 추론을 수행하는 스크립트입니다. `train.py`와 동일한 데이터 구조와 전처리 파이프라인을 사용하며, `model.py`의 이중 모드 구조를 활용하여 두 가지 예측 방식을 제공합니다.

### 주요 기능

#### 1. **이중 예측 모드**
- **RL State 모드**: 강화학습 에이전트용 상태 벡터 생성 (기본값)
- **Forecast 모드**: 미래 시계열 데이터 예측

#### 2. **자동 모델 로딩**
- 최신 훈련된 모델 자동 감지 및 로드
- 타임스탬프 기반 모델 버전 관리
- GPU/CPU 자동 감지 및 최적화

#### 3. **동일한 전처리 파이프라인**
- `train.py`와 일관된 데이터 처리
- 종목별 MinMaxScaler 적용
- 88개 특성 (81 TA + 7 News) 구조 유지

### 핵심 함수 분석

#### `load_latest_model(model_dir, model_config, device)`
```python
def load_latest_model(model_dir: str, model_config: dict, device: torch.device):
    # 최신 모델 파일 자동 감지
    model_pattern = os.path.join(model_dir, "tst_model_best_*.pt")
    model_files = glob.glob(model_pattern)
    latest_model_path = max(model_files, key=os.path.getmtime)
    
    # 모델 초기화 및 가중치 로드
    model = TSTModel(config_dict=model_config).to(device)
    checkpoint = torch.load(latest_model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
```

**특징**:
- `tst_model_best_*.pt` 패턴으로 훈련된 모델 검색
- 파일 수정 시간 기준으로 최신 모델 선택
- 모델 파라미터 수 출력으로 로딩 검증

#### `prepare_data_for_prediction(data_path, target_ticker, context_length)`
```python
def prepare_data_for_prediction(data_path: str, target_ticker: str = None, context_length: int = 60):
    # 1. TA 데이터 로드 (train.py와 동일)
    ta_history_df = pd.read_csv(data_path)
    ta_history_df.set_index(['Date', 'Ticker'], inplace=True)
    
    # 2. 특정 종목 필터링 (선택사항)
    if target_ticker:
        ta_history_df = ta_history_df.xs(target_ticker, level='Ticker', drop_level=False)
    
    # 3. 중립 뉴스 데이터 생성
    neutral_news_df = create_neutral_news_df(ta_history_df.index, NEWS_FEATURE_COLS)
    
    # 4. 특성 결합 및 스케일링
    combined_features_df = align_and_combine_features(ta_history_df, neutral_news_df)
    scaled_features_df = apply_per_ticker_scaling(combined_features_df)
```

**중요 사항**:
- `train.py`와 정확히 동일한 전처리 순서
- 종목별 독립적 스케일링 유지
- 최소 `context_length` 일수 데이터 검증

#### `create_prediction_sequences(scaled_data, context_length)`
```python
def create_prediction_sequences(scaled_data: pd.DataFrame, context_length: int):
    for ticker, group in scaled_data.groupby(level='Ticker'):
        ticker_data = group.values
        # 최신 context_length개 데이터포인트 추출
        latest_sequence = ticker_data[-context_length:]  # (60, 88)
        
        prediction_data[ticker] = {
            'sequence': torch.FloatTensor(latest_sequence).unsqueeze(0),  # (1, 60, 88)
            'dates': latest_dates,
            'last_date': latest_dates[-1]
        }
```

**핵심 로직**:
- 각 종목의 **가장 최근 60일** 데이터 사용
- 배치 차원 추가 (`unsqueeze(0)`)
- 예측 기준일 메타데이터 보존

#### `predict_with_tst_model(model, prediction_data, device, mode)`

**RL State 모드** (기본값):
```python
if mode == 'rl_state':
    model.eval()  # Inference 모드
    rl_state = model(past_values=sequence)  # (1, 256)
    # 강화학습 에이전트용 상태 벡터 반환
```

**Forecast 모드**:
```python
elif mode == 'forecast':
    model.train()  # Training 모드 (임시)
    outputs = model(past_values=sequence)
    # 시계열 예측 결과 추출
    forecast = outputs.params[0]  # (1, 10, 88)
```

### 사용 방법

#### 명령행 인터페이스

```bash
# 기본 사용법 (모든 종목, RL State 모드)
python tst_model/predict.py

# 특정 종목 예측
python tst_model/predict.py --ticker AAPL

# 시계열 예측 모드
python tst_model/predict.py --mode forecast

# 경로 지정
python tst_model/predict.py \
    --model_dir ./custom_models \
    --data_path ./custom_data.csv \
    --output_dir ./predictions
```

#### 프로그래매틱 사용

```python
from tst_model.predict import load_latest_model, prepare_data_for_prediction, predict_with_tst_model

# 모델 로드
model, model_path = load_latest_model("./tst_model_output", DEFAULT_MODEL_CONFIG, device)

# 데이터 준비
data_info = prepare_data_for_prediction("./data.csv", target_ticker="AAPL")

# 예측 수행
predictions = predict_with_tst_model(model, prediction_data, device, mode='rl_state')
```

### 출력 결과

#### RL State 모드 출력
```
tst_predictions/
├── AAPL_rl_state_20231215_143052.npy     # (256,) 벡터
├── GOOGL_rl_state_20231215_143052.npy    # (256,) 벡터
└── prediction_summary_20231215_143052.txt # 요약 정보
```

**RL State 벡터 특성**:
- **차원**: (rl_state_size,) = (256,)
- **용도**: RL 에이전트의 입력 상태
- **내용**: TST 모델이 추출한 고차원 시장 특성

#### Forecast 모드 출력
```
tst_predictions/
├── AAPL_forecast_20231215_143052.csv     # (10, 88) 테이블
├── GOOGL_forecast_20231215_143052.csv    # (10, 88) 테이블
└── prediction_summary_20231215_143052.txt # 요약 정보
```

**Forecast 데이터 구조**:
- **차원**: (prediction_length, input_size) = (10, 88)
- **내용**: 향후 10일간 88개 특성 예측값
- **형식**: CSV 파일 (prediction_day 인덱스)

### 모델 통합 구조

#### 데이터 흐름
```
Historical Data (60일) → 전처리 → 스케일링 → TST Model → 예측 결과
     ↓                     ↓           ↓           ↓
88개 특성 → 중립 뉴스 결합 → MinMax → Inference → RL State/Forecast
```

#### TST 모델 활용
```python
# model.py의 forward 메소드 활용
if self.training:
    return transformer_output  # Loss 포함 (Forecast 모드)
else:
    return rl_state           # RL 상태 벡터 (RL State 모드)
```

### 특징 및 장점

#### 1. **Production-Ready 설계**
- 명령행 인터페이스로 운영 환경 통합 용이
- 에러 핸들링 및 상세한 로깅
- 타임스탬프 기반 결과 파일 버전 관리

#### 2. **유연한 예측 모드**
- **RL 통합**: 강화학습 시스템과 직접 연동
- **시계열 분석**: 전통적인 예측 분석 지원
- **배치 처리**: 여러 종목 동시 처리

#### 3. **일관성 보장**
- 훈련과 추론 간 동일한 전처리 파이프라인
- 스케일링 방식 및 특성 순서 일치
- 모델 구조 및 하이퍼파라미터 동기화

#### 4. **메타데이터 관리**
- 예측 기준일 추적
- 사용된 모델 경로 기록
- 결과 통계 요약

### 시스템 내 역할

이 예측 스크립트는 주식 트레이딩 봇에서 다음과 같은 핵심 역할을 수행합니다:

1. **RL 에이전트 공급**: 최신 시장 데이터를 RL 상태 벡터로 변환
2. **시장 예측**: 단기 시계열 전망 제공
3. **실시간 추론**: 새로운 데이터에 대한 즉시 예측
4. **의사결정 지원**: 정량적 시장 분석 결과 제공

이를 통해 **훈련된 TST 모델**이 실제 트레이딩 환경에서 **실시간 시장 분석**과 **RL 에이전트 상태 생성**을 담당하게 됩니다.

---

## TST Model VRAM 사용량 분석

### 모델 파라미터 기반 메모리 계산

#### 현재 모델 설정
```python
DEFAULT_MODEL_CONFIG = {
    'input_size': 88,           # 81 TA + 7 News features
    'prediction_length': 10,    # 10일 예측
    'context_length': 60,       # 60일 히스토리
    'n_layer': 3,              # 3 encoder + 3 decoder layers
    'n_head': 4,               # 4 attention heads
    'd_model': 128,            # 128 차원 트랜스포머
    'rl_state_size': 256,      # 256 차원 RL 상태 벡터
    'batch_size': 32           # 배치 크기 (훈련 시)
}
```

### 파라미터 수 계산

#### 1. **TimeSeriesTransformer 구성요소**

**Embedding Layers**:
```
- Input Embedding: 88 × 128 = 11,264 params
- Position Embedding: 60 × 128 = 7,680 params
- Total Embedding: ~19K params
```

**Encoder Layers (3개)**:
```
각 레이어당:
- Multi-Head Attention: 4 × (128 × 128 × 3) + bias = ~197K params
- Layer Norm: 128 × 2 = 256 params  
- Feed Forward: 128 × 512 + 512 × 128 + bias = ~131K params
- Layer Norm: 128 params

레이어당 총합: ~328K params
3개 레이어: ~984K params
```

**Decoder Layers (3개)**:
```
각 레이어당:
- Self-Attention: ~197K params
- Cross-Attention: ~197K params  
- Feed Forward: ~131K params
- Layer Norms: ~400 params

레이어당 총합: ~525K params
3개 레이어: ~1.6M params
```

**Output Projection**:
```
- Final Linear: 128 × 88 = 11,264 params
- Distribution params (mean, std): 추가 파라미터
```

#### 2. **RL Head**
```
- Linear Layer: (10 × 88) × 256 + 256 bias = 225,536 params
```

#### 3. **총 파라미터 수 추정**
```
- TimeSeriesTransformer: ~2.8M params
- RL Head: ~225K params
- Total: ~3.0M params
```

### 메모리 사용량 계산

#### **추론 시 (Inference)**

**모델 파라미터**:
```
3,000,000 params × 4 bytes (float32) = 12 MB
```

**입력 데이터** (단일 종목):
```
Batch × Context × Features × 4 bytes
= 1 × 60 × 88 × 4 = 21,120 bytes ≈ 21 KB
```

**중간 활성화 메모리**:
```
- Attention 행렬: 4 heads × 60 × 60 × 4 bytes = 57.6 KB
- Hidden states: 60 × 128 × 4 bytes = 30.7 KB  
- Feed forward: 60 × 512 × 4 bytes = 122.8 KB
- 기타 중간 계산: ~200 KB

총 중간 활성화: ~411 KB
```

**RL State 출력**:
```
1 × 256 × 4 bytes = 1,024 bytes ≈ 1 KB
```

**추론 시 총 VRAM**: **약 13-15 MB**

#### **훈련 시 (Training)**

**모델 파라미터**: 12 MB (동일)

**배치 입력 데이터**:
```
배치 × (과거 + 미래) × 특성 × 4 bytes
= 32 × (60 + 10) × 88 × 4 = 787,456 bytes ≈ 787 KB
```

**그래디언트 메모리**:
```
파라미터와 동일한 크기: 12 MB
```

**옵티마이저 상태 (AdamW)**:
```
- Momentum: 12 MB
- Variance: 12 MB  
- Total: 24 MB
```

**중간 활성화 (배치별)**:
```
- Attention: 32 × 4 × 60 × 60 × 4 = 1.8 MB
- Hidden states: 32 × 60 × 128 × 4 = 983 KB
- Feed forward: 32 × 60 × 512 × 4 = 3.9 MB
- 백워드 패스용 추가 메모리: ~5 MB

총 활성화: ~12 MB
```

**훈련 시 총 VRAM**: **약 60-70 MB**

### 실제 VRAM 사용량 측정 코드

```python
import torch
import psutil
import GPUtil

def measure_model_memory():
    """TST 모델의 실제 메모리 사용량 측정"""
    
    # GPU 메모리 측정 (CUDA 사용 시)
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.cuda.empty_cache()
        
        # 초기 GPU 메모리
        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
        
        # 모델 로드
        model = TSTModel(DEFAULT_MODEL_CONFIG).to(device)
        model_memory = torch.cuda.memory_allocated() / 1024**2 - initial_memory
        
        # 추론 테스트
        dummy_input = torch.randn(1, 60, 88).to(device)
        model.eval()
        with torch.no_grad():
            _ = model(dummy_input)
        inference_memory = torch.cuda.memory_allocated() / 1024**2 - initial_memory
        
        # 훈련 테스트
        model.train()
        dummy_target = torch.randn(1, 10, 88).to(device)
        output = model(dummy_input, dummy_target)
        loss = output.loss
        loss.backward()
        training_memory = torch.cuda.memory_allocated() / 1024**2 - initial_memory
        
        return {
            'model_params_mb': model_memory,
            'inference_mb': inference_memory, 
            'training_mb': training_memory,
            'total_params': sum(p.numel() for p in model.parameters())
        }
    
    else:
        # CPU 메모리 측정
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024**2
        
        model = TSTModel(DEFAULT_MODEL_CONFIG)
        model_memory = process.memory_info().rss / 1024**2 - initial_memory
        
        return {
            'model_params_mb': model_memory,
            'cpu_memory': True,
            'total_params': sum(p.numel() for p in model.parameters())
        }

# 사용 예시
print("=== TST Model Memory Usage ===")
memory_stats = measure_model_memory()
for key, value in memory_stats.items():
    print(f"{key}: {value}")
```

### 메모리 최적화 방안

#### 1. **추론 최적화**
```python
# Mixed Precision 사용
model = model.half()  # float16 사용 시 메모리 50% 절약

# 배치 크기 조정
PREDICT_CONFIG['batch_size'] = 1  # 실시간 추론 시
```

#### 2. **훈련 최적화**
```python
# Gradient Accumulation
effective_batch_size = 32
actual_batch_size = 8  # 메모리에 맞게 조정
accumulation_steps = effective_batch_size // actual_batch_size

# Mixed Precision Training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    output = model(past_values, future_values)
    loss = output.loss / accumulation_steps
scaler.scale(loss).backward()
```

#### 3. **메모리 모니터링**
```python
# 실시간 메모리 추적
def log_memory_usage(stage_name):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"{stage_name} - Allocated: {allocated:.1f}MB, Cached: {cached:.1f}MB")
```

### 하드웨어 권장사항

#### **최소 요구사항**
- **GPU**: 2GB VRAM (GTX 1060, RTX 3050 급)
- **용도**: 추론 전용, 작은 배치 크기

#### **권장 사양**
- **GPU**: 4-6GB VRAM (RTX 3060, RTX 4060 급)  
- **용도**: 효율적인 훈련 및 추론

#### **최적 환경**
- **GPU**: 8GB+ VRAM (RTX 3070, RTX 4070 급)
- **용도**: 대용량 배치, 다중 모델 실험

### 실제 벤치마크 결과 (예상)

| 작업 모드 | 배치 크기 | VRAM 사용량 | 처리 속도 |
|-----------|-----------|-------------|-----------|
| 추론 (단일) | 1 | ~15MB | ~50ms |
| 추론 (배치) | 32 | ~45MB | ~200ms |
| 훈련 (소형) | 8 | ~35MB | ~500ms |
| 훈련 (표준) | 32 | ~70MB | ~1500ms |

이 분석을 통해 **대부분의 현대적인 GPU에서 무리 없이 실행 가능**하며, 특히 추론 작업은 매우 가벼운 메모리 요구사항을 가지고 있음을 알 수 있습니다.

---




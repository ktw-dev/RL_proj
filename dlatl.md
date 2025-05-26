
## API 키 및 환경 설정 가이드

### 🔐 보안 설정 (API Keys Configuration)

이 프로젝트는 여러 외부 API를 사용하므로 API 키를 안전하게 관리하는 것이 중요합니다. 환경변수를 사용하여 민감한 정보를 git repository에서 분리했습니다.

#### **1. 초기 설정 (First Time Setup)**

```bash
# 1. 프로젝트 클론
git clone <repository-url>
cd reinforcement_project-main

# 2. 의존성 설치
pip install -r requirements.txt

# 3. 환경변수 파일 생성
cp .env.example .env

# 4. API 키 설정 (다음 단계 참조)
nano .env  # 또는 원하는 에디터 사용
```

#### **2. 필수 API 키 획득**

| API | 용도 | 무료 한도 | 획득 방법 |
|-----|------|-----------|-----------|
| **Alpha Vantage** | 주식 OHLCV 데이터 | 500 calls/day | [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key) |
| **News API** | 뉴스 헤드라인 수집 | 1000 requests/day | [https://newsapi.org/register](https://newsapi.org/register) |
| **Quandl** (선택) | 추가 금융 데이터 | 50 calls/day | [https://www.quandl.com/](https://www.quandl.com/) |

#### **3. .env 파일 설정**

`.env` 파일을 생성하고 다음과 같이 설정하세요:

```bash
# ===== API Keys =====
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_key_here
NEWS_API_KEY=your_news_api_key_here
QUANDL_API_KEY=your_quandl_key_here

# ===== Model Configuration =====
DEFAULT_BATCH_SIZE=32
DEFAULT_CONTEXT_LENGTH=60
DEFAULT_PREDICTION_LENGTH=10

# ===== Trading Configuration =====
SUPPORTED_TICKERS=AAPL,GOOGL,MSFT,AMZN,TSLA,NVDA,META,NFLX
DEFAULT_INVESTMENT_AMOUNT=10000.0

# ===== Debug Mode =====
DEBUG_MODE=False
LOG_LEVEL=INFO
```

#### **4. 설정 검증**

```python
# 설정이 올바른지 확인
python -c "from config.settings import validate_api_keys; validate_api_keys()"

# 출력 예시:
# ✅ All required API keys are configured
```

#### **5. Git 저장소 보안**

```bash
# .env 파일이 git에 추가되지 않았는지 확인
git status

# 다음이 표시되어야 함:
# On branch main
# nothing to commit, working tree clean

# .env 파일은 .gitignore에 의해 자동으로 제외됨
```

### 🏗️ 프로젝트 구조 및 설정 파일

```
reinforcement_project-main/
├── config/
│   ├── settings.py              # 환경변수 기반 설정 (git에 포함)
│   └── settings_template.py     # 설정 템플릿 (참고용)
├── .env                         # 실제 API 키들 (git에서 제외)
├── .env.example                 # 환경변수 템플릿 (git에 포함)
├── .gitignore                   # 민감한 파일들 제외 설정
└── requirements.txt             # python-dotenv 포함
```

### 🔧 고급 설정 옵션

#### **환경별 설정**

```bash
# 개발 환경
DEBUG_MODE=True
LOG_LEVEL=DEBUG

# 프로덕션 환경
DEBUG_MODE=False
LOG_LEVEL=INFO
```

#### **성능 튜닝**

```bash
# GPU 메모리가 부족한 경우
DEFAULT_BATCH_SIZE=16
DEFAULT_CONTEXT_LENGTH=30

# 고성능 GPU 사용 시
DEFAULT_BATCH_SIZE=64
DEFAULT_CONTEXT_LENGTH=120
```

#### **커스텀 경로 설정**

```bash
# 모델 저장 경로 변경
TST_MODEL_PATH=/custom/path/to/models/
RL_AGENT_MODEL_PATH=/custom/path/to/agents/

# 데이터 디렉토리 변경
DATA_DIR=/large/storage/data/
TST_PREDICTIONS_DIR=/fast/ssd/predictions/
```

### 🚨 보안 주의사항

#### **❌ 절대 하지 말 것**
```bash
# API 키를 코드에 직접 하드코딩
ALPHA_VANTAGE_API_KEY = "abc123"  # 위험!

# .env 파일을 git에 커밋
git add .env  # 절대 금지!
```

#### **✅ 권장 사항**
```bash
# 환경변수 사용
export ALPHA_VANTAGE_API_KEY="your_key"
python your_script.py

# .env 파일 권한 설정 (Linux/Mac)
chmod 600 .env

# API 키 정기 교체 (보안 강화)
```

### 🔍 트러블슈팅

#### **API 키 에러**
```python
# 에러: ⚠️ Configuration Warning: Missing required API keys: ALPHA_VANTAGE_API_KEY

# 해결방법:
# 1. .env 파일 존재 확인
# 2. API 키 값 확인 (공백, 따옴표 제거)
# 3. 파일 권한 확인
```

#### **import 에러**
```bash
# 에러: ModuleNotFoundError: No module named 'dotenv'

# 해결방법:
pip install python-dotenv
```

#### **API 한도 초과**
```python
# Alpha Vantage: 500 calls/day 제한
# News API: 1000 requests/day 제한

# 해결방법:
# 1. 여러 API 키 로테이션 사용
# 2. 캐싱 구현으로 API 호출 최소화
# 3. 프리미엄 플랜 업그레이드 고려
```

### 📋 설정 체크리스트

- [ ] `.env` 파일 생성 완료
- [ ] 모든 필수 API 키 설정 완료
- [ ] `python-dotenv` 설치 완료
- [ ] 설정 검증 성공 (`validate_api_keys()`)
- [ ] `.env` 파일이 git에 추가되지 않음 확인
- [ ] 프로젝트 실행 테스트 완료

이 설정을 완료하면 API 키를 안전하게 관리하면서 프로젝트를 git repository에 공유할 수 있습니다.

'''
/train.py --max_samples 129723 --epochs 100
Traceback (most recent call last):
  File "/home/theta/Public/reinforcement_project-main/rl_agent/train.py", line 3, in <module>
    import torch
ModuleNotFoundError: No module named 'torch'
theta@theta-14700k-4080:~/Public/reinforcement_project-main$ conda activate rl_project
(rl_project) theta@theta-14700k-4080:~/Public/reinforcement_project-main$ python3 rl_agent/train.py --max_samples 129723 --epochs 100
=== TST-RL Training Workflow ===
Agent type: PPO
Training mode: Multi-ticker
🚀 GPU Available: 1 device(s)
   Current GPU: NVIDIA GeForce RTX 4080
   GPU Memory: 15.7 GB
   CUDA Version: 12.6
Max tickers: 20
Min samples per ticker: 100
TST model directory: /home/theta/Public/reinforcement_project-main/tst_model_output
Data path: /home/theta/Public/reinforcement_project-main/all_tickers_historical_features.csv
Training epochs: 100
Output directory: /home/theta/Public/reinforcement_project-main/rl_model_output
=== Training PPO Agent on Multiple Tickers ===
=== Generating RL States from TST Model ===
Target ticker: All tickers
Model directory: /home/theta/Public/reinforcement_project-main/tst_model_output
Data path: /home/theta/Public/reinforcement_project-main/all_tickers_historical_features.csv
Device: cuda
Loading historical data from: /home/theta/Public/reinforcement_project-main/all_tickers_historical_features.csv
Loaded TA data. Shape: (130923, 80)
Tickers found: ['AAPL', 'AMD', 'AMZN', 'ASML', 'AVGO', 'AZN', 'COST', 'CSCO', 'GOOG', 'GOOGL', 'INTU', 'ISRG', 'LIN', 'META', 'MSFT', 'NFLX', 'NVDA', 'PLTR', 'TMUS', 'TSLA']
Found 80 numeric TA features
Adding synthetic neutral news sentiment features...
Total features after adding news sentiment: 87 (80 TA + 7 news)
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
Loading model from: /home/theta/Public/reinforcement_project-main/tst_model_output/tst_model_best_20250524_191519.pt
Model loaded successfully. Parameters: 1,848,279
Loaded TST model: /home/theta/Public/reinforcement_project-main/tst_model_output/tst_model_best_20250524_191519.pt
Using close prices for AAPL
Generated 10883 RL states for AAPL
Using close prices for AMD
Generated 11071 RL states for AMD
Using close prices for AMZN
Generated 6731 RL states for AMZN
Using close prices for ASML
Generated 7280 RL states for ASML
Using close prices for AVGO
Generated 3655 RL states for AVGO
Using close prices for AZN
Generated 7745 RL states for AZN
Using close prices for COST
Generated 9476 RL states for COST
Using close prices for CSCO
Generated 8562 RL states for CSCO
Using close prices for GOOG
Generated 4905 RL states for GOOG
Using close prices for GOOGL
Generated 4905 RL states for GOOGL
Using close prices for INTU
Generated 7787 RL states for INTU
Using close prices for ISRG
Generated 5952 RL states for ISRG
Using close prices for LIN
Generated 7973 RL states for LIN
Using close prices for META
Generated 2953 RL states for META
Using close prices for MSFT
Generated 9557 RL states for MSFT
Using close prices for NFLX
Generated 5469 RL states for NFLX
Using close prices for NVDA
Generated 6306 RL states for NVDA
Using close prices for PLTR
Generated 848 RL states for PLTR
Using close prices for TMUS
Generated 4235 RL states for TMUS
Using close prices for TSLA
Generated 3430 RL states for TSLA
Selected 20 tickers for training: ['AAPL', 'AMD', 'AMZN', 'ASML', 'AVGO', 'AZN', 'COST', 'CSCO', 'GOOG', 'GOOGL', 'INTU', 'ISRG', 'LIN', 'META', 'MSFT', 'NFLX', 'NVDA', 'PLTR', 'TMUS', 'TSLA']
  AAPL: 10883 samples, price range: $0.04-$196.67, mean: $20.60
  AMD: 11071 samples, price range: $1.62-$211.38, mean: $19.51
  AMZN: 6731 samples, price range: $0.10-$189.50, mean: $36.46
  ASML: 7280 samples, price range: $2.01-$1037.87, mean: $126.46
  AVGO: 3655 samples, price range: $1.03-$138.41, mean: $23.69
  AZN: 7745 samples, price range: $1.61-$76.05, mean: $19.75
  COST: 9476 samples, price range: $4.04-$780.30, mean: $95.37
  CSCO: 8562 samples, price range: $0.05-$57.49, mean: $18.20
  GOOG: 4905 samples, price range: $4.09-$172.87, mean: $44.07
  GOOGL: 4905 samples, price range: $4.11-$171.13, mean: $44.13
  INTU: 7787 samples, price range: $1.94-$679.06, mean: $102.67
  ISRG: 5952 samples, price range: $0.74-$400.59, mean: $87.90
  LIN: 7973 samples, price range: $4.19-$468.99, mean: $86.73
  META: 2953 samples, price range: $17.65-$525.42, mean: $168.70
  MSFT: 9557 samples, price range: $0.06-$425.34, mean: $50.55
  NFLX: 5469 samples, price range: $0.37-$691.69, mean: $141.11
  NVDA: 6306 samples, price range: $0.03-$94.97, mean: $5.27
  PLTR: 848 samples, price range: $6.00-$39.00, mean: $16.48
  TMUS: 4235 samples, price range: $8.96-$165.09, mean: $59.35
  TSLA: 3430 samples, price range: $1.30-$409.97, mean: $74.74

Combined training data:
  Total samples: 129723
  RL state shape: (129723, 256)
  Price range: $0.03 - $1037.87
  Price mean: $59.05
  Data shuffled for better training

Starting PPO training...
Using device for RL training: cuda
Training RL agent on device: cuda
Using all 129723 samples for training
PPO networks moved to cuda
Epoch 1/100 | Total Loss: 0.0790 | Policy: 0.0114 | Value: 0.6751 | GPU: 0.33GB used, 1.21GB cached
Epoch 2/100 | Total Loss: 0.1014 | Policy: 0.0022 | Value: 0.9920 | GPU: 0.33GB used, 1.47GB cached
Epoch 3/100 | Total Loss: 0.0984 | Policy: 0.0044 | Value: 0.9395 | GPU: 0.33GB used, 1.47GB cached
Epoch 4/100 | Total Loss: 0.0932 | Policy: 0.0069 | Value: 0.8634 | GPU: 0.33GB used, 1.47GB cached
Epoch 5/100 | Total Loss: 0.1003 | Policy: 0.0002 | Value: 1.0006 | GPU: 0.33GB used, 1.47GB cached
Epoch 6/100 | Total Loss: 0.1026 | Policy: 0.0040 | Value: 0.9862 | GPU: 0.33GB used, 1.47GB cached
Epoch 7/100 | Total Loss: 0.1005 | Policy: 0.0005 | Value: 0.9997 | GPU: 0.33GB used, 1.47GB cached
Epoch 8/100 | Total Loss: 0.0991 | Policy: 0.0022 | Value: 0.9691 | GPU: 0.33GB used, 1.47GB cached
Epoch 9/100 | Total Loss: 0.1004 | Policy: 0.0057 | Value: 0.9477 | GPU: 0.33GB used, 1.47GB cached
...

```
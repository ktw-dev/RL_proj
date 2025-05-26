
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

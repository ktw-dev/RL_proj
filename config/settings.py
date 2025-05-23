# config/settings.py: Stores API keys, model paths, and other global constants.

# API Keys (replace with your actual keys or load from environment variables)
ALPHA_VANTAGE_API_KEY = "56N9KNLCC7LIWPLS"
QUANDL_API_KEY = "sgRpg71_paTR1ufW7D3Z"
NEWS_API_KEY = "5abb6bde86e94334a6a4d5c2a2eee095"

# Model Paths
TST_MODEL_PATH = "models/tst/tst_model.pth"
RL_AGENT_MODEL_PATH = "models/rl/rl_agent.zip"
FINBERT_MODEL_PATH = "ProsusAI/finbert" # or local path if downloaded

# Other settings
DATA_DIR = "data/"
RAW_DATA_DIR = "data/raw/"
PROCESSED_DATA_DIR = "data/processed/" 
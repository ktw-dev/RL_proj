# config/settings_template.py: Template for settings with environment variables
# Copy this file to settings.py and configure your API keys

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# API Keys (loaded from environment variables)
# Set these in your .env file or system environment
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', '')
QUANDL_API_KEY = os.getenv('QUANDL_API_KEY', '')
NEWS_API_KEY = os.getenv('NEWS_API_KEY', '')

# Model Paths (can be overridden by environment variables)
TST_MODEL_PATH = os.getenv('TST_MODEL_PATH', 'tst_model_output/')
RL_AGENT_MODEL_PATH = os.getenv('RL_AGENT_MODEL_PATH', 'rl_model_output/')
FINBERT_MODEL_PATH = os.getenv('FINBERT_MODEL_PATH', 'ProsusAI/finbert')

# Data Directories
DATA_DIR = os.getenv('DATA_DIR', 'data/')
RAW_DATA_DIR = os.getenv('RAW_DATA_DIR', 'data/raw/')
PROCESSED_DATA_DIR = os.getenv('PROCESSED_DATA_DIR', 'data/processed/')

# Output Directories  
TST_PREDICTIONS_DIR = os.getenv('TST_PREDICTIONS_DIR', 'tst_predictions/')
LOGS_DIR = os.getenv('LOGS_DIR', 'logs/')

# Model Configuration
DEFAULT_BATCH_SIZE = int(os.getenv('DEFAULT_BATCH_SIZE', '32'))
DEFAULT_CONTEXT_LENGTH = int(os.getenv('DEFAULT_CONTEXT_LENGTH', '60'))
DEFAULT_PREDICTION_LENGTH = int(os.getenv('DEFAULT_PREDICTION_LENGTH', '10'))

# Trading Configuration
SUPPORTED_TICKERS = os.getenv('SUPPORTED_TICKERS', 'AAPL,GOOGL,MSFT,AMZN,TSLA').split(',')
DEFAULT_INVESTMENT_AMOUNT = float(os.getenv('DEFAULT_INVESTMENT_AMOUNT', '10000.0'))

# API Configuration
MAX_RETRIES = int(os.getenv('MAX_RETRIES', '3'))
API_TIMEOUT = int(os.getenv('API_TIMEOUT', '30'))

# Debug and Logging
DEBUG_MODE = os.getenv('DEBUG_MODE', 'False').lower() == 'true'
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# Validation function
def validate_api_keys():
    """Validate that all required API keys are set."""
    missing_keys = []
    
    if not ALPHA_VANTAGE_API_KEY:
        missing_keys.append('ALPHA_VANTAGE_API_KEY')
    if not NEWS_API_KEY:
        missing_keys.append('NEWS_API_KEY')
    # QUANDL_API_KEY is optional for this project
    
    if missing_keys:
        raise ValueError(f"Missing required API keys: {', '.join(missing_keys)}")
    
    print("✅ All required API keys are configured")
    return True

# Auto-validate on import (can be disabled by setting SKIP_VALIDATION=True)
if not os.getenv('SKIP_VALIDATION', 'False').lower() == 'true':
    try:
        validate_api_keys()
    except ValueError as e:
        print(f"⚠️  Configuration Warning: {e}")
        print("📝 Please check your .env file or environment variables") 
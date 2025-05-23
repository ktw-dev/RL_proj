# data_collection/news_analyzer.py: Fetches news and performs sentiment analysis.
import feedparser
import pandas as pd
from datetime import datetime, timedelta, timezone
import time
import urllib.parse
import requests # For NewsAPI
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import sys
from pathlib import Path

# Add parent directory for config import
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from config import settings # For API keys and model path
from config import tickers as cfg_tickers # For company names, supported tickers

# --- Constants ---
# Google News RSS
BASE_GOOGLE_NEWS_URL = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"
RSS_REQUEST_DELAY = 1  # seconds

# NewsAPI
NEWS_API_KEY = settings.NEWS_API_KEY
NEWS_API_EVERYTHING_URL = "https://newsapi.org/v2/everything"
NEWS_API_REQUEST_DELAY = 1 
MAX_HEADLINES_PER_DAY_PER_TICKER = 10

# FinBERT
FINBERT_MODEL_NAME = settings.FINBERT_MODEL_PATH
TOKENIZER = None
MODEL = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- FinBERT Model Loading ---
def load_finbert_model():
    """Loads the FinBERT model and tokenizer globally."""
    global TOKENIZER, MODEL
    if TOKENIZER is None or MODEL is None:
        print(f"Loading FinBERT model ('{FINBERT_MODEL_NAME}') and tokenizer on device: {DEVICE}...")
        try:
            TOKENIZER = AutoTokenizer.from_pretrained(FINBERT_MODEL_NAME)
            MODEL = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL_NAME)
            MODEL.to(DEVICE)
            MODEL.eval()
            print("FinBERT model and tokenizer loaded successfully.")
        except Exception as e:
            print(f"Error loading FinBERT model: {e}")
            TOKENIZER, MODEL = None, None # Ensure they remain None if loading fails
            raise # Re-raise to signal critical failure

load_finbert_model() # Load on module import

# --- Sentiment Analysis ---
def analyze_sentiment_headlines(headlines: list):
    """Analyzes sentiment of a list of headlines using FinBERT."""
    if MODEL is None or TOKENIZER is None:
        print("Error: FinBERT model not loaded. Cannot analyze sentiment.")
        return [] # Return empty list if model not available

    if not headlines:
        return []

    inputs = TOKENIZER(headlines, padding=True, truncation=True, return_tensors="pt", max_length=256) # Max length for headlines
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = MODEL(**inputs)
        predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    
    results = []
    id2label = {0: "positive", 1: "negative", 2: "neutral"} # Standard for ProsusAI/finbert
    for i in range(predictions.shape[0]):
        prob_dict = {label_name: predictions[i, j].item() for j, label_name in id2label.items()}
        results.append(prob_dict)
    return results

# --- News Fetching Helpers (adapted from news_fetcher.py) ---
def _normalize_date_to_utc_string(date_obj):
    if isinstance(date_obj, str):
        try:
            date_obj = datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
        except ValueError:
            date_obj = datetime.strptime(date_obj, '%Y-%m-%d')
    if date_obj.tzinfo is None:
        date_obj = date_obj.replace(tzinfo=timezone.utc)
    else:
        date_obj = date_obj.astimezone(timezone.utc)
    return date_obj.isoformat()

def _clean_headline(headline):
    return headline.strip() if isinstance(headline, str) else ""

def _make_google_query_string(ticker, start_date_str, end_date_str):
    # Google News 'when:Xd' is less reliable for specific ranges.
    # Using 'after:' and 'before:' for more precision.
    query_parts = [f'"{cfg_tickers.COMPANY_NAMES.get(ticker, ticker)}" OR "{ticker}" stock price']
    query_parts.append(f"after:{start_date_str}")
    query_parts.append(f"before:{end_date_str}")
    return urllib.parse.quote_plus(" ".join(query_parts))

def _parse_rss_feed_for_ticker(ticker_symbol, query_string, start_datetime_utc, end_datetime_utc):
    news_items = []
    url = BASE_GOOGLE_NEWS_URL.format(query=query_string)
    try:
        feed = feedparser.parse(url)
        if feed.bozo: print(f"Warning (RSS): Feed for {ticker_symbol} may be malformed. Error: {feed.bozo_exception}")

        for entry in feed.entries:
            headline = getattr(entry, 'title', 'N/A')
            if headline == 'N/A': continue

            published_time_struct = getattr(entry, 'published_parsed', None)
            if not published_time_struct: continue
            
            published_datetime_utc = datetime.fromtimestamp(time.mktime(published_time_struct), tz=timezone.utc)

            if published_datetime_utc < start_datetime_utc or published_datetime_utc > end_datetime_utc:
                continue
                
            news_items.append({
                'ticker': ticker_symbol, 'headline': _clean_headline(headline),
                'published_datetime_utc': published_datetime_utc,
                'link': getattr(entry, 'link', 'N/A'),
                'source_name': getattr(entry.get('source'), 'title', 'Google News RSS'),
                'fetch_source': 'RSS'
            })
    except Exception as e:
        print(f"Error (RSS) fetching/parsing for {ticker_symbol}: {e}")
    time.sleep(RSS_REQUEST_DELAY)
    return news_items

def _fetch_newsapi_articles(ticker_symbol, start_date_str_utc, end_date_str_utc):
    news_items = []
    if not NEWS_API_KEY or NEWS_API_KEY == "YOUR_NEWS_API_KEY":
        print("Warning (NewsAPI): API key not configured. Skipping NewsAPI fetch.")
        return news_items

    headers = {'X-Api-Key': NEWS_API_KEY}
    # Use company name if available for broader search, along with ticker
    company_name = cfg_tickers.COMPANY_NAMES.get(ticker_symbol, ticker_symbol)
    query = f'("{ticker_symbol}" OR "{company_name}") AND (stock OR shares OR equity OR market)'
    
    params = {
        'q': query, 'language': 'en', 'sortBy': 'publishedAt', 'pageSize': 100, # Max page size
        'from': _normalize_date_to_utc_string(start_date_str_utc),
        'to': _normalize_date_to_utc_string(end_date_str_utc)
    }
    print(f"Fetching NewsAPI for {ticker_symbol} with query '{query}' from {params['from']} to {params['to']}")
    try:
        response = requests.get(NEWS_API_EVERYTHING_URL, headers=headers, params=params)
        response.raise_for_status()
        articles = response.json().get('articles', [])

        for article in articles:
            headline = article.get('title')
            if not headline or headline == "[Removed]": continue

            published_at_str = article.get('publishedAt')
            if not published_at_str: continue
            published_datetime_utc = datetime.fromisoformat(published_at_str.replace('Z', '+00:00')).astimezone(timezone.utc)
            
            news_items.append({
                'ticker': ticker_symbol, 'headline': _clean_headline(headline),
                'published_datetime_utc': published_datetime_utc,
                'link': article.get('url', 'N/A'),
                'source_name': article.get('source', {}).get('name', 'NewsAPI'),
                'fetch_source': 'NewsAPI'
            })
    except requests.exceptions.RequestException as e:
        print(f"Error (NewsAPI) for {ticker_symbol}: {e}")
    except Exception as e:
        print(f"Unexpected NewsAPI error for {ticker_symbol}: {e}")
    time.sleep(NEWS_API_REQUEST_DELAY)
    return news_items

# --- Main Public Function ---
def fetch_and_analyze_recent_news(ticker_symbol: str, analysis_base_date_utc: datetime = None):
    """
    Fetches news for a ticker for up to 7 previous business days + subsequent weekend days
    relative to analysis_base_date_utc, and performs sentiment analysis. Limits to 10 unique headlines per day.
    Dates are handled in UTC. User in Korea (KST) should provide analysis_base_date converted to UTC.
    If analysis_base_date_utc is None, it defaults to datetime.now(timezone.utc).

    Args:
        ticker_symbol (str): The stock ticker.
        analysis_base_date_utc (datetime, optional): The UTC datetime for which the analysis is being run. 
                                                    News will be fetched for the period *before* this date.
                                                    Defaults to datetime.now(timezone.utc).

    Returns:
        pd.DataFrame: Columns: ['ticker', 'published_date', 'published_time_utc', 'headline', 
                                'sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 
                                'link', 'source_name', 'fetch_source']
                      Sorted by published_datetime_utc.
    """
    if MODEL is None or TOKENIZER is None:
        print("FinBERT model not loaded. Aborting news analysis.")
        return pd.DataFrame()

    if analysis_base_date_utc is None:
        analysis_base_date_utc = datetime.now(timezone.utc)
    elif analysis_base_date_utc.tzinfo is None: # Ensure UTC
        analysis_base_date_utc = analysis_base_date_utc.replace(tzinfo=timezone.utc)
    else:
        analysis_base_date_utc = analysis_base_date_utc.astimezone(timezone.utc)

    print(f"Starting news analysis for {ticker_symbol} based on UTC date: {analysis_base_date_utc.strftime('%Y-%m-%d')}")

    # Determine the date range: 7 business days prior + subsequent weekend
    # We need to find the date that was 7 business days *before* analysis_base_date_utc's date part
    current_check_date = analysis_base_date_utc.date()
    business_days_to_count = 7
    earliest_fetch_date_limit = current_check_date
    
    days_counted = 0
    temp_date = current_check_date
    while days_counted < business_days_to_count:
        temp_date -= timedelta(days=1)
        if temp_date.weekday() < 5: # Monday to Friday (0-4)
            days_counted += 1
    earliest_fetch_date_limit = temp_date # This is the 7th business day

    # The fetch window ends on the day *before* the analysis_base_date_utc's date part.
    # News *up to* analysis_base_date_utc (exclusive of its date, for "previous" days)
    # So, if analysis_base_date is Tue, end_date for fetch is Mon.
    fetch_end_date_inclusive = analysis_base_date_utc.date() - timedelta(days=1)
    
    # Ensure fetch_start_date is not after fetch_end_date_inclusive
    fetch_start_date = min(earliest_fetch_date_limit, fetch_end_date_inclusive)

    if fetch_start_date > fetch_end_date_inclusive:
        print(f"Calculated start date {fetch_start_date} is after end date {fetch_end_date_inclusive}. No news to fetch.")
        return pd.DataFrame()

    # Convert to datetime for fetching functions, ensuring full days are covered
    # NewsAPI and RSS helpers expect string dates YYYY-MM-DD for their API calls.
    # The actual datetime filtering for RSS will use start_datetime_utc and end_datetime_utc
    start_datetime_utc = datetime(fetch_start_date.year, fetch_start_date.month, fetch_start_date.day, 0, 0, 0, tzinfo=timezone.utc)
    end_datetime_utc = datetime(fetch_end_date_inclusive.year, fetch_end_date_inclusive.month, fetch_end_date_inclusive.day, 23, 59, 59, tzinfo=timezone.utc)

    print(f"Fetching news for {ticker_symbol} from {start_datetime_utc.strftime('%Y-%m-%d %H:%M:%S %Z')} to {end_datetime_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")

    # Fetch from sources
    rss_query_str = _make_google_query_string(ticker_symbol, start_datetime_utc.strftime('%Y-%m-%d'), end_datetime_utc.strftime('%Y-%m-%d'))
    rss_articles = _parse_rss_feed_for_ticker(ticker_symbol, rss_query_str, start_datetime_utc, end_datetime_utc)
    
    newsapi_articles = _fetch_newsapi_articles(ticker_symbol, start_datetime_utc.strftime('%Y-%m-%d'), end_datetime_utc.strftime('%Y-%m-%d'))
    
    all_articles_raw = rss_articles + newsapi_articles
    if not all_articles_raw:
        print(f"No raw news articles found for {ticker_symbol} in the period.")
        return pd.DataFrame()

    # Create DataFrame and filter by MAX_HEADLINES_PER_DAY_PER_TICKER
    raw_df = pd.DataFrame(all_articles_raw)
    raw_df['published_datetime_utc'] = pd.to_datetime(raw_df['published_datetime_utc'], utc=True)
    raw_df.sort_values(by='published_datetime_utc', ascending=False, inplace=True) # Get latest first

    # Deduplicate headlines (simplified: exact match on headline text for a given day)
    # More sophisticated deduplication might consider similarity.
    raw_df['published_date_str'] = raw_df['published_datetime_utc'].dt.strftime('%Y-%m-%d')
    raw_df.drop_duplicates(subset=['ticker', 'headline', 'published_date_str'], keep='first', inplace=True)
    
    # Apply MAX_HEADLINES_PER_DAY_PER_TICKER
    # Group by date, then take top N headlines for that date.
    # Since already sorted by published_datetime_utc descending, head(N) should give latest within that day.
    final_articles_list = []
    for date_str, group in raw_df.groupby('published_date_str'):
        final_articles_list.extend(group.head(MAX_HEADLINES_PER_DAY_PER_TICKER).to_dict('records'))
    
    if not final_articles_list:
        print(f"No articles remaining after daily limit for {ticker_symbol}.")
        return pd.DataFrame()

    headlines_df = pd.DataFrame(final_articles_list)
    headlines_df.sort_values(by='published_datetime_utc', ascending=True, inplace=True) # Sort chronologically for output

    # Perform sentiment analysis
    headlines_to_analyze = headlines_df['headline'].tolist()
    sentiment_scores_list = analyze_sentiment_headlines(headlines_to_analyze)

    if not sentiment_scores_list or len(sentiment_scores_list) != len(headlines_df):
        print(f"Sentiment analysis returned inconsistent results for {ticker_symbol}. Expected {len(headlines_df)} scores, got {len(sentiment_scores_list)}.")
        # Fallback: return df without sentiment scores or with NaNs
        for col in ['sentiment_positive', 'sentiment_negative', 'sentiment_neutral']: headlines_df[col] = np.nan
        return headlines_df[['ticker', 'published_datetime_utc', 'headline', 'link', 'source_name', 'fetch_source', 
                             'sentiment_positive', 'sentiment_negative', 'sentiment_neutral']]


    sentiment_df = pd.DataFrame(sentiment_scores_list)
    # Merge sentiment scores back into the headlines_df
    # Ensure index alignment for direct assignment or use join/merge
    headlines_df = pd.concat([headlines_df.reset_index(drop=True), sentiment_df.reset_index(drop=True)], axis=1)

    # Prepare final columns
    headlines_df['published_date'] = headlines_df['published_datetime_utc'].dt.date
    headlines_df['published_time_utc'] = headlines_df['published_datetime_utc'].dt.strftime('%H:%M:%S')
    
    # Ensure all expected columns exist
    output_cols = ['ticker', 'published_date', 'published_time_utc', 'published_datetime_utc', 
                   'headline', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral', 
                   'link', 'source_name', 'fetch_source']
    for col in output_cols:
        if col not in headlines_df.columns:
            headlines_df[col] = np.nan # Add missing columns with NaN if any step failed to produce them

    return headlines_df[output_cols]


if __name__ == '__main__':
    # Ensure COMPANY_NAMES is available for NewsAPI query robustness
    if not hasattr(cfg_tickers, 'COMPANY_NAMES'):
        print("Note: cfg_tickers.COMPANY_NAMES not defined. NewsAPI query may be less effective.")
        cfg_tickers.COMPANY_NAMES = {}

    example_ticker = "AAPL" 
    # Test with a specific analysis date (in UTC)
    # KST current time: datetime.now(timezone(timedelta(hours=9)))
    # Corresponding UTC: datetime.now(timezone.utc)
    
    # If today is Wed, 7 business days ago is last Tue. 
    # News from last Tue, Wed, Thu, Fri, Mon, Tue, (today-1)Wed.
    # analysis_date_utc_test = datetime.now(timezone.utc)
    # For reproducible test, let's fix a date, e.g., a Wednesday
    analysis_date_utc_test = datetime(2023, 10, 25, 10, 0, 0, tzinfo=timezone.utc) # A Wednesday
    print(f"--- Testing fetch_and_analyze_recent_news for {example_ticker} based on UTC: {analysis_date_utc_test.strftime('%Y-%m-%d')} ---")
    
    # What this means:
    # Analysis for Wed 2023-10-25. We need news *before* this date.
    # Fetch end date: 2023-10-24 (Tue)
    # 7 business days before Tue 2023-10-24:
    # 1. Mon 10-23
    # 2. Fri 10-20
    # 3. Thu 10-19
    # 4. Wed 10-18
    # 5. Tue 10-17
    # 6. Mon 10-16
    # 7. Fri 10-13
    # So, earliest fetch date is Fri 2023-10-13.
    # Fetch window: 2023-10-13 00:00:00 UTC to 2023-10-24 23:59:59 UTC
    
    analyzed_news_df = fetch_and_analyze_recent_news(example_ticker, analysis_base_date_utc=analysis_date_utc_test)

    if not analyzed_news_df.empty:
        print(f"\nFetched and analyzed {len(analyzed_news_df)} news articles for {example_ticker}.")
        print("Columns:", analyzed_news_df.columns.tolist())
        print("Sample data (last 5 rows):")
        print(analyzed_news_df.tail())
        print(f"\nDate range in results: {analyzed_news_df['published_datetime_utc'].min()} to {analyzed_news_df['published_datetime_utc'].max()}")
        
        # Check daily headline count
        daily_counts = analyzed_news_df.groupby(analyzed_news_df['published_datetime_utc'].dt.date)['headline'].count()
        print("\nDaily headline counts:")
        print(daily_counts)
        assert daily_counts.max() <= MAX_HEADLINES_PER_DAY_PER_TICKER, "Exceeded max headlines per day"

    else:
        print(f"No news analyzed for {example_ticker}.")

    print(f"\n--- Test with another ticker, e.g., MSFT, and default analysis_base_date (now) ---")
    # example_ticker_2 = "MSFT"
    # analyzed_news_df_2 = fetch_and_analyze_recent_news(example_ticker_2) # Uses datetime.now(timezone.utc)
    # if not analyzed_news_df_2.empty:
    #     print(f"\nFetched and analyzed {len(analyzed_news_df_2)} news articles for {example_ticker_2}.")
    #     print(analyzed_news_df_2[['published_date', 'headline', 'sentiment_positive']].tail())
    # else:
    #     print(f"No news analyzed for {example_ticker_2}.")
    print("Note: Test with MSFT using current time is commented out to ensure consistent test runs with fixed date.") 
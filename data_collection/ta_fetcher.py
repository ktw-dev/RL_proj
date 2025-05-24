# data_collection/ta_fetcher.py: Functions to fetch OHLCV and other technical data from yfinance, AlphaVantage, and Quandl. 
# 05-24-2025 19:28

import yfinance as yf
from alpha_vantage.timeseries import TimeSeries
import quandl
import pandas as pd
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to allow imports from sibling directories
current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent
sys.path.append(str(parent_dir))

from config import settings
from config.tickers import SUPPORTED_TICKERS
import time

# Import for the new main block
try:
    from feature_engineering.ta_calculator import calculate_technical_indicators
except ImportError as e:
    print(f"Could not import calculate_technical_indicators: {e}")
    print("Ensure feature_engineering.ta_calculator is accessible.")
    # Define a placeholder if import fails, so the rest of the script can be parsed,
    # but the main block relying on it will fail informatively.
    calculate_technical_indicators = None


# Alpha Vantage and Quandl API keys from settings
ALPHA_VANTAGE_API_KEY = settings.ALPHA_VANTAGE_API_KEY
QUANDL_API_KEY = settings.QUANDL_API_KEY

# Configure Quandl API key
if QUANDL_API_KEY and QUANDL_API_KEY != "YOUR_QUANDL_API_KEY":
    quandl.ApiConfig.api_key = QUANDL_API_KEY
else:
    print("Warning: Quandl API key not configured or is placeholder. Quandl fetching will be limited or fail.")

# --- yfinance Fetcher ---
def fetch_yfinance_data(ticker_symbol, start_date_str, end_date_str, interval="1d"):
    """
    Fetches historical market data from Yahoo Finance.
    :param ticker_symbol: Stock ticker symbol (e.g., "AAPL")
    :param start_date_str: Start date string in 'YYYY-MM-DD' format
    :param end_date_str: End date string in 'YYYY-MM-DD' format
    :param interval: Data interval (e.g., "1d", "1h", "1wk")
    :return: pandas DataFrame with OHLCV data, or empty DataFrame on error.
    """
    print(f"Fetching yfinance data for {ticker_symbol} from {start_date_str} to {end_date_str}")
    try:
        ticker = yf.Ticker(ticker_symbol)
        data = ticker.history(start=start_date_str, end=end_date_str, interval=interval)
        if data.empty:
            print(f"No yfinance data found for {ticker_symbol} in the given range.")
            return pd.DataFrame()
        data.reset_index(inplace=True)
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
        elif 'Datetime' in data.columns: # Handle intraday data key
            data['Datetime'] = pd.to_datetime(data['Datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
            # Rename 'Datetime' to 'Date' for consistency if it's daily data in disguise or for processing
            # This might need adjustment based on how intraday data is later processed.
            # For now, if interval suggests daily, we prefer 'Date'.
            if interval == "1d" or interval.endswith("d"):
                 data.rename(columns={'Datetime': 'Date'}, inplace=True)
                 if 'Date' in data.columns: # Ensure format if renamed
                      data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')

        # Ensure index is DatetimeIndex for consistent processing later if 'Date' column is primary
        if 'Date' in data.columns and not isinstance(data.index, pd.DatetimeIndex):
            try:
                data_date_col = data['Date'] # Store before potential drop if set_index fails
                data = data.set_index(pd.DatetimeIndex(data['Date']))
                if 'Date' in data.columns: # drop if it became index successfully
                    data.drop(columns=['Date'], inplace=True, errors='ignore')
                data.reset_index(inplace=True) # And bring it back as a column
            except Exception as e:
                print(f"Warning: Could not set DatetimeIndex from Date column in yfinance: {e}")
                data['Date'] = data_date_col # Restore if failed

        return data
    except Exception as e:
        print(f"Error fetching yfinance data for {ticker_symbol}: {e}")
        return pd.DataFrame()

# --- Alpha Vantage Fetcher ---
def fetch_alphavantage_data(ticker_symbol, start_date_str, end_date_str, outputsize='full'):
    """
    Fetches historical daily adjusted stock data from Alpha Vantage.
    Note: Free tier has limitations (e.g., 5 calls per minute, 500 per day, limited history for some functions).
    :param ticker_symbol: Stock ticker symbol
    :param start_date_str: Start date string in 'YYYY-MM-DD' format
    :param end_date_str: End date string in 'YYYY-MM-DD' format
    :param outputsize: 'compact' (last 100 data points) or 'full' (full-length history)
    :return: pandas DataFrame with daily adjusted data, or empty DataFrame on error.
    """
    if not ALPHA_VANTAGE_API_KEY or ALPHA_VANTAGE_API_KEY == "YOUR_ALPHA_VANTAGE_API_KEY":
        print("Warning: Alpha Vantage API key not configured or is placeholder. Skipping Alpha Vantage fetch.")
        return pd.DataFrame()
    
    print(f"Fetching Alpha Vantage data for {ticker_symbol} (Output: {outputsize}) from {start_date_str} to {end_date_str}")
    ts = TimeSeries(key=ALPHA_VANTAGE_API_KEY, output_format='pandas')
    try:
        data, meta_data = ts.get_daily_adjusted(symbol=ticker_symbol, outputsize=outputsize)
        data = data.sort_index(ascending=True) 
        data.reset_index(inplace=True)
        data.rename(columns={'date': 'Date', 
                             '1. open': 'Open', '2. high': 'High', '3. low': 'Low', 
                             '4. close': 'Close', '5. adjusted close': 'Adj Close', 
                             '6. volume': 'Volume'}, inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
        data = data[(data['Date'] >= start_date_str) & (data['Date'] <= end_date_str)]
        if data.empty:
            print(f"No Alpha Vantage data found for {ticker_symbol} in the given range after filtering.")
        return data[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]
    except Exception as e:
        print(f"Error fetching Alpha Vantage data for {ticker_symbol}: {e}")
        # Add rate limit wait here as a common point of failure/retry
        print("Waiting 13 seconds due to Alpha Vantage potential rate limit or error.")
        time.sleep(13) 
        return pd.DataFrame()
    finally:
        # Ensure to wait even if successful, to respect overall limits if making multiple calls.
        time.sleep(13)

# --- Quandl Fetcher ---
def fetch_quandl_data(ticker_symbol, start_date_str, end_date_str, quandl_db_code="EOD"):
    """
    Fetches historical stock data from Quandl (End of Day US Stock Prices - EOD dataset recommended for free tier).
    Note: Many premium datasets exist. WIKI dataset was free but is no longer updated.
          Availability and dataset codes can vary. Example uses generic EOD/{ticker}.
    :param ticker_symbol: Stock ticker symbol
    :param start_date_str: Start date string in 'YYYY-MM-DD' format
    :param end_date_str: End date string in 'YYYY-MM-DD' format
    :param quandl_db_code: The Quandl database code to use (e.g., "EOD", "WIKI/PRICES" - though WIKI is old)
    :return: pandas DataFrame with data, or empty DataFrame on error.
    """
    if not QUANDL_API_KEY or QUANDL_API_KEY == "YOUR_QUANDL_API_KEY":
        print("Warning: Quandl API key not configured or is placeholder. Skipping Quandl fetch.")
        return pd.DataFrame()

    dataset_code = f"{quandl_db_code}/{ticker_symbol.upper()}" # Ensure ticker is uppercase for EOD
    print(f"Fetching Quandl data for {dataset_code} from {start_date_str} to {end_date_str}")
    try:
        data = quandl.get(dataset_code, start_date=start_date_str, end_date=end_date_str)
        if data.empty:
            print(f"No Quandl data found for {dataset_code} in the given range.")
            return pd.DataFrame()
        data.reset_index(inplace=True)
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
        # Standardize column names for EOD dataset (common ones)
        rename_map = {
            'Adj_Open': 'Open', # Prefer 'Open' if 'Adj_Open' is the primary open
            'Adj_High': 'High',
            'Adj_Low': 'Low',
            'Adj_Close': 'Close', # Prefer 'Close' if 'Adj_Close' is the primary close
            'Adj_Volume': 'Volume',
            # Keep original names if they exist and are standard
            'Open': 'Open', 'High': 'High', 'Low': 'Low', 'Close': 'Close', 'Volume': 'Volume'
        }
        # Filter rename_map to only include columns present in data
        actual_rename_map = {k: v for k, v in rename_map.items() if k in data.columns}
        data.rename(columns=actual_rename_map, inplace=True)

        # Ensure essential columns are present, otherwise it's not useful
        # This list aligns with yfinance and Alpha Vantage primary columns after rename
        essential_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in essential_cols):
            print(f"Quandl data for {dataset_code} is missing some essential columns after renaming. Required: {essential_cols}, Got: {data.columns.tolist()}")
            # Attempt to fill missing standard OHLCV from adjusted if originals are missing
            if 'Open' not in data.columns and 'Adj. Open' in data.columns: data.rename(columns={'Adj. Open': 'Open'}, inplace=True)
            if 'High' not in data.columns and 'Adj. High' in data.columns: data.rename(columns={'Adj. High': 'High'}, inplace=True)
            if 'Low' not in data.columns and 'Adj. Low' in data.columns: data.rename(columns={'Adj. Low': 'Low'}, inplace=True)
            if 'Close' not in data.columns and 'Adj. Close' in data.columns: data.rename(columns={'Adj. Close': 'Close'}, inplace=True)
            if 'Volume' not in data.columns and 'Adj. Volume' in data.columns: data.rename(columns={'Adj. Volume': 'Volume'}, inplace=True)

            # Recheck
            if not all(col in data.columns for col in essential_cols):
                print(f"Still missing essential columns for Quandl data {dataset_code}. Returning empty.")
                return pd.DataFrame()
        
        return data[essential_cols] # Return only standardized essential columns
    except Exception as e:
        # Handle various Quandl errors (including NotFoundError, AuthenticationError, etc.)
        error_msg = str(e).lower()
        if 'not found' in error_msg or '404' in error_msg:
            print(f"Quandl dataset {dataset_code} not found. Check ticker and database code.")
        elif 'authentication' in error_msg or 'api key' in error_msg:
            print(f"Quandl authentication error for {dataset_code}. Check API key.")
        else:
            print(f"Error fetching Quandl data for {dataset_code}: {e}")
        return pd.DataFrame()

# --- Main Fetching Orchestrator (Historical) ---
def fetch_all_historical_ta(ticker_symbol, start_date_str, end_date_str):
    """
    Fetches historical TA data from all available sources for a single ticker and date range.
    This function demonstrates fetching; actual merging/priority should be handled in feature engineering.
    
    :param ticker_symbol: Stock ticker.
    :param start_date_str: Start date 'YYYY-MM-DD'.
    :param end_date_str: End date 'YYYY-MM-DD'.
    :return: Dictionary of DataFrames: {'yfinance': df_yf, 'alphavantage': df_av, 'quandl': df_qdl}
    """
    print(f"\nFetching all TA data for {ticker_symbol} from {start_date_str} to {end_date_str}...")
    
    # yfinance
    df_yf = fetch_yfinance_data(ticker_symbol, start_date_str, end_date_str)
    
    # Alpha Vantage
    df_av = fetch_alphavantage_data(ticker_symbol, start_date_str, end_date_str)
    
    # Quandl (using EOD as example, might require subscription or different code for some tickers)
    df_qdl = fetch_quandl_data(ticker_symbol, start_date_str, end_date_str, quandl_db_code="EOD")
    
    all_data = {
        'yfinance': df_yf,
        'alphavantage': df_av,
        'quandl': df_qdl
    }
    
    # Basic check on retrieved data
    for source, df in all_data.items():
        if not df.empty:
            print(f"Successfully fetched data from {source} for {ticker_symbol}: {len(df)} rows.")
        else:
            print(f"No data fetched from {source} for {ticker_symbol}.")
            
    return all_data

# --- Fetcher for Real-time Inference Needs (Prioritizing AlphaVantage, Quandl) ---
def fetch_recent_ohlcv_for_inference(ticker_symbol, business_days=14, lookback_period=60):
    """
    Fetches recent OHLCV data, prioritizing Alpha Vantage, then Quandl, then yfinance.
    It fetches data for `business_days` ending today, plus an additional `lookback_period`.
    :param ticker_symbol: Stock ticker symbol (e.g., "AAPL")
    :param business_days: Number of recent business days of data to focus on (default: 14).
    :param lookback_period: Additional number of past days needed for TA calculations (e.g., 50-60 days).
    :return: pandas DataFrame with OHLCV data, or empty DataFrame on error.
    """
    print(f"Fetching recent OHLCV data for {ticker_symbol} for inference (Target: last {business_days} business days + {lookback_period} days lookback)...")
    
    ohlcv_df = pd.DataFrame()
    source_used = "None"

    # Determine date range for fetching
    end_date = datetime.now()
    # Estimate calendar days needed: (business_days for final TAs + lookback_period for calculation) * safety_factor_for_weekends_holidays
    # For example, if we need TAs for 14 biz days, and TAs need 60 days lookback, total effective days ~74.
    # Fetching more calendar days initially is safer.
    # Let's keep the previous logic for calendar day estimation for fetching window
    calendar_days_for_biz_output = business_days * 2 # Heuristic for business days to calendar days
    total_calendar_days_to_fetch = calendar_days_for_biz_output + lookback_period + 30 # Generous buffer

    start_date_dt = end_date - timedelta(days=total_calendar_days_to_fetch)
    start_date_str = start_date_dt.strftime('%Y-%m-%d')
    # For end_date_str, yfinance and Quandl are usually fine with today's date (they give up to previous close).
    # Alpha Vantage also usually fine.
    end_date_str = end_date.strftime('%Y-%m-%d')

    min_required_rows = business_days + lookback_period # Ideal minimum if all are trading days
    # A more realistic minimum might be (business_days + lookback_period) * 0.6 if many non-trading days
    realistic_min_rows = int((business_days + lookback_period) * 0.7)


    # 1. Try yfinance first (most reliable and free)
    print(f"Attempting to fetch data from yfinance for {ticker_symbol}...")
    ohlcv_df = fetch_yfinance_data(ticker_symbol, start_date_str, end_date_str)
    if not ohlcv_df.empty and len(ohlcv_df) >= realistic_min_rows:
        source_used = "yfinance"
        print(f"Successfully fetched sufficient data from yfinance for {ticker_symbol}.")
    else:
        print(f"yfinance data for {ticker_symbol} was insufficient (got {len(ohlcv_df)}, need ~{realistic_min_rows}). Trying other sources...")
        ohlcv_df = pd.DataFrame() # Reset if insufficient

    # 2. Try Alpha Vantage if yfinance failed (use compact for free tier)
    if ohlcv_df.empty:
        if ALPHA_VANTAGE_API_KEY and ALPHA_VANTAGE_API_KEY != "YOUR_ALPHA_VANTAGE_API_KEY":
            print(f"Attempting to fetch data from Alpha Vantage for {ticker_symbol}...")
            # Use 'compact' for free tier (last 100 data points)
            ohlcv_df = fetch_alphavantage_data(ticker_symbol, start_date_str, end_date_str, outputsize='compact')
            if not ohlcv_df.empty and len(ohlcv_df) >= min(realistic_min_rows, 90):  # Adjust expectation for compact
                source_used = "Alpha Vantage"
                print(f"Successfully fetched sufficient data from Alpha Vantage for {ticker_symbol}.")
            else:
                print(f"Alpha Vantage data for {ticker_symbol} was empty or insufficient (got {len(ohlcv_df)}, need ~{realistic_min_rows}).")
                ohlcv_df = pd.DataFrame() # Reset if insufficient
        else:
            print("Alpha Vantage API key not configured. Skipping.")

    # 3. Try Quandl if both failed (with error handling fix)
    if ohlcv_df.empty:
        if QUANDL_API_KEY and QUANDL_API_KEY != "YOUR_QUANDL_API_KEY":
            print(f"Attempting to fetch data from Quandl (EOD) for {ticker_symbol}...")
            ohlcv_df = fetch_quandl_data(ticker_symbol, start_date_str, end_date_str, quandl_db_code="EOD")
            if not ohlcv_df.empty and len(ohlcv_df) >= realistic_min_rows:
                source_used = "Quandl"
                print(f"Successfully fetched sufficient data from Quandl for {ticker_symbol}.")
            else:
                print(f"Quandl data for {ticker_symbol} was empty or insufficient (got {len(ohlcv_df)}, need ~{realistic_min_rows}).")
                ohlcv_df = pd.DataFrame() # Reset if insufficient
        else:
            print("Quandl API key not configured. Skipping.")
            
    # Final processing if data was successfully fetched from any source
    if not ohlcv_df.empty:
        print(f"Data fetched from {source_used}. Processing {len(ohlcv_df)} rows...")
        
        # Ensure 'Date' column exists and is string, set as DatetimeIndex for filtering
        if 'Date' not in ohlcv_df.columns and isinstance(ohlcv_df.index, pd.DatetimeIndex):
             ohlcv_df.reset_index(inplace=True) # Make Date a column if it's the index
        
        if 'Date' in ohlcv_df.columns:
            ohlcv_df['Date'] = pd.to_datetime(ohlcv_df['Date'])
            ohlcv_df.set_index('Date', inplace=True)
        elif isinstance(ohlcv_df.index, pd.DatetimeIndex):
            pass # Already has DatetimeIndex
        else:
            print("Error: DataFrame has no 'Date' column or DatetimeIndex. Cannot proceed with date filtering.")
            return pd.DataFrame()

        ohlcv_df.sort_index(ascending=True, inplace=True)
        
        # Filter to get the last `business_days` of data PLUS the `lookback_period`
        # This logic aims to get enough data for TA calculation for the last N business days.
        # The actual number of rows needed is lookback_period + N (target business days)
        # The `tail` method after ensuring enough history can get us the most recent portion.
        
        # We need to ensure we have at least `lookback_period` rows before the last `business_days`
        # A simpler way: fetch a broad window, then take the tail(lookback_period + business_days_as_rows)
        # Estimate business_days_as_rows (e.g., business_days * 1.0 for safety, assuming dense data)
        
        # The most recent `business_days` in the index gives the target end dates
        # We need data from (earliest_target_date - lookback_period) to latest_target_date
        
        # Prune to ensure we don't have excessively old data if fetch window was too large
        # This should give approximately lookback_period + N_business_days_worth_of_rows
        # where N_business_days_worth_of_rows is likely more than 'business_days' due to weekends.
        # Let's aim for about `lookback_period + business_days (target) + some_buffer_for_weekends` rows in final output.
        # e.g. lookback 60, business_days 14 -> we need about 74 trading days.
        
        # Take the most recent (lookback_period + business_days + buffer_for_non_trading_days) trading days
        # business_days parameter is for the final *output* of TAs. For fetching, we need more for lookback.
        num_rows_to_keep = lookback_period + int(business_days * 1.5) # approx calendar days for business days
        
        if len(ohlcv_df) > num_rows_to_keep:
            filtered_df = ohlcv_df.tail(num_rows_to_keep).copy()
        else:
            filtered_df = ohlcv_df.copy()

        if len(filtered_df) < realistic_min_rows : # Check again after tailing
             print(f"Warning: After filtering, data might be insufficient. Source: {source_used}. Got {len(filtered_df)} rows, target for TA calc: {realistic_min_rows}")
        
        filtered_df.reset_index(inplace=True) # Bring 'Date' back as a column
        filtered_df['Date'] = pd.to_datetime(filtered_df['Date']).dt.strftime('%Y-%m-%d')
        
        print(f"Successfully processed data from {source_used} for {ticker_symbol}. Shape: {filtered_df.shape}")
        if not filtered_df.empty:
             print(f"Final date range for {ticker_symbol}: {filtered_df['Date'].iloc[0]} to {filtered_df['Date'].iloc[-1]}")
        return filtered_df
    else:
        print(f"Could not fetch recent OHLCV data for {ticker_symbol} from any source.")
        return pd.DataFrame()

if __name__ == '__main__':
    if calculate_technical_indicators is None:
        print("Aborting due to missing 'calculate_technical_indicators' function.")
    else:
        # Define the ticker to process. It must be in SUPPORTED_TICKERS.
        # You can change this to any ticker from your config/tickers.py
        target_ticker = "AAPL" 

        if target_ticker not in SUPPORTED_TICKERS:
            print(f"Error: Ticker '{target_ticker}' is not in the SUPPORTED_TICKERS list.")
            print(f"Supported tickers are: {SUPPORTED_TICKERS}")
        else:
            print(f"--- Processing last 30 days of data for {target_ticker} ---")

            # Define date range: last 30 days from today
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            start_date_str = start_date.strftime('%Y-%m-%d')
            end_date_str = end_date.strftime('%Y-%m-%d')

            print(f"Fetching OHLCV data for {target_ticker} from {start_date_str} to {end_date_str}")
            
            # fetch_yfinance_data is expected to return OHLCV with 'Date' as a column.
            ohlcv_df = fetch_yfinance_data(target_ticker, start_date_str, end_date_str, interval="1d")

            if not ohlcv_df.empty:
                print(f"Successfully fetched OHLCV data. Shape: {ohlcv_df.shape}")
                
                # Prepare DataFrame for calculate_technical_indicators:
                # Ensure 'Date' column is pd.to_datetime and set as DatetimeIndex.
                # This is crucial for consistency with how ta_fetcher_history.py prepares data
                # for the same TA calculation function.
                if 'Date' in ohlcv_df.columns:
                    try:
                        ohlcv_df['Date'] = pd.to_datetime(ohlcv_df['Date'])
                        ohlcv_df.set_index('Date', inplace=True)
                        print("Successfully set 'Date' column as DatetimeIndex.")
                    except Exception as e:
                        print(f"Error processing 'Date' column for {target_ticker}: {e}")
                        ohlcv_df = pd.DataFrame() # Invalidate df if date processing fails
                else:
                    print(f"Error: 'Date' column missing in OHLCV data for {target_ticker}.")
                    ohlcv_df = pd.DataFrame() # Invalidate df

                if not ohlcv_df.empty:
                    # Calculate technical indicators using the imported function
                    print(f"Calculating technical indicators for {target_ticker}...")
                    # Pass ohlcv_df.copy() as done in ta_fetcher_history.py
                    features_df = calculate_technical_indicators(ohlcv_df.copy(), include_ohlcv=True)
                    
                    if not features_df.empty:
                        print(f"Successfully calculated technical indicators. Shape: {features_df.shape}")
                        
                        # Save to CSV. The resulting CSV will have Date as index (if preserved by TA calc)
                        # and OHLCV + TAs as columns.
                        csv_filename = f"{target_ticker}_last_30days_features.csv"
                        try:
                            features_df.to_csv(csv_filename, index=True) # Save with index (Date)
                            print(f"Saved data for {target_ticker} to {csv_filename}")
                            print(f"Output file preview (head):\n{features_df.head()}")
                        except Exception as e:
                            print(f"Error saving data to CSV for {target_ticker}: {e}")
                    else:
                        print(f"Failed to calculate technical indicators for {target_ticker}.")
            else:
                print(f"No OHLCV data found for {target_ticker} in the range {start_date_str} to {end_date_str}.")

            print(f"\n--- Processing for {target_ticker} finished. ---")
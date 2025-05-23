# feature_engineering/ta_fetcher_history.py: Fetches all historical OHLCV data from listing date 
# and calculates all technical indicators as defined in ta_calculator.py

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add the project root to the Python path to enable absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Now we can use absolute imports that work from anywhere
from data_collection.ta_fetcher import fetch_yfinance_data # Using yfinance as primary source for history
from feature_engineering.ta_calculator import calculate_technical_indicators
from config.tickers import SUPPORTED_TICKERS # Import SUPPORTED_TICKERS
from config import settings

DEFAULT_VERY_EARLY_START_DATE = "1970-01-01"

def get_ticker_first_trade_date_str(ticker_symbol):
    """
    Attempts to find the first trade date for a ticker using yfinance.
    Returns date string in 'YYYY-MM-DD' format or a default early date if not found.
    """
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        info = ticker_obj.info
        first_trade_epoch = info.get('firstTradeDateEpochUtc')
        if first_trade_epoch:
            # Convert from UTC epoch seconds to datetime, then to YYYY-MM-DD string
            first_trade_date = datetime.fromtimestamp(first_trade_epoch, datetime.UTC).strftime('%Y-%m-%d')
            print(f"First trade date for {ticker_symbol} from yf.info: {first_trade_date}")
            return first_trade_date
        else:
            print(f"No 'firstTradeDateEpochUtc' in yf.info for {ticker_symbol}. Using default early date.")
            # Attempt to get the earliest data point from history if info fails
            hist = ticker_obj.history(period="max")
            if not hist.empty:
                return hist.index[0].strftime('%Y-%m-%d')
            return DEFAULT_VERY_EARLY_START_DATE
    except Exception as e:
        print(f"Error getting first trade date for {ticker_symbol} from yfinance: {e}. Using default early date.")
        return DEFAULT_VERY_EARLY_START_DATE

def fetch_ohlcv_history_for_ticker(ticker_symbol, start_date_str, end_date_str):
    """
    Fetches OHLCV data for a given ticker and date range using primarily yfinance.
    """
    print(f"Fetching historical OHLCV for {ticker_symbol} from {start_date_str} to {end_date_str}")
    # Using fetch_yfinance_data from ta_fetcher.py for consistency in data format
    ohlcv_df = fetch_yfinance_data(ticker_symbol, start_date_str, end_date_str, interval="1d")
    
    if ohlcv_df.empty:
        print(f"Could not fetch historical OHLCV data for {ticker_symbol} between {start_date_str} and {end_date_str}.")
    return ohlcv_df

def fetch_and_process_historical_data(ticker_symbol, global_end_date_str="2025-05-09"):
    """
    Fetches all historical OHLCV data for a ticker from its listing date up to global_end_date_str,
    then calculates all technical indicators based on 'extended_custom_strategy' via
    `calculate_technical_indicators`.

    :param ticker_symbol: Stock ticker symbol (e.g., "AAPL")
    :param global_end_date_str: The final end date for fetching data ('YYYY-MM-DD').
    :return: pandas DataFrame with OHLCV and all calculated technical indicators, or empty DataFrame on error.
    """
    print(f"--- Starting historical data processing for {ticker_symbol} up to {global_end_date_str} ---")
    
    start_date_str = get_ticker_first_trade_date_str(ticker_symbol)
    
    # Ensure start_date is not after global_end_date_str
    if pd.to_datetime(start_date_str) > pd.to_datetime(global_end_date_str):
        print(f"Ticker {ticker_symbol}'s first trade date ({start_date_str}) is after the global end date ({global_end_date_str}). No data to fetch.")
        return pd.DataFrame()

    ohlcv_df = fetch_ohlcv_history_for_ticker(ticker_symbol, start_date_str, global_end_date_str)

    if ohlcv_df.empty:
        print(f"No OHLCV data found for {ticker_symbol}. Skipping TA calculation.")
        return pd.DataFrame()
    
    print(f"OHLCV data fetched for {ticker_symbol}. Shape: {ohlcv_df.shape}. Now calculating TAs...")
    
    # Calculate technical indicators using the function from ta_calculator.py
    # This function internally uses the extended_custom_strategy and handles NaN filling.
    features_df = calculate_technical_indicators(ohlcv_df.copy(), include_ohlcv=True)
    
    if features_df.empty:
        print(f"Failed to calculate technical indicators for {ticker_symbol}.")
        return pd.DataFrame()
        
    print(f"Successfully calculated technical indicators for {ticker_symbol}. Final shape: {features_df.shape}")
    print(f"--- Finished historical data processing for {ticker_symbol} ---")
    return features_df

if __name__ == '__main__':
    # Use SUPPORTED_TICKERS from config/tickers.py
    # test_tickers = ["AAPL"] # Add more tickers as needed
    # test_tickers = ["NONEXISTENTTICKER"] # Test non-existent ticker
    # test_tickers = ["BRK-A"] # Test ticker with potentially different info structure if needed
    
    all_historical_data_list = [] # Changed from dict to list to store DataFrames

    #for ticker in SUPPORTED_TICKERS: # Changed to use SUPPORTED_TICKERS
    for ticker in SUPPORTED_TICKERS: # Changed to use SUPPORTED_TICKERS
        print(f"\nProcessing ticker: {ticker}")
        historical_data_with_tas = fetch_and_process_historical_data(ticker, global_end_date_str="2024-05-10") # Using a recent past date for example
        
        if not historical_data_with_tas.empty:
            historical_data_with_tas['Ticker'] = ticker # Add Ticker column
            all_historical_data_list.append(historical_data_with_tas) # Append DataFrame to list
            print(f"Data for {ticker} (head):\n{historical_data_with_tas.head()}")
            print(f"Data for {ticker} (tail):\n{historical_data_with_tas.tail()}")
            print(f"Number of features for {ticker}: {len(historical_data_with_tas.columns)}")
            # Individual CSV saving removed
            # csv_filename = f"{ticker}_historical_features.csv"
            # historical_data_with_tas.to_csv(csv_filename)
            # print(f"Saved data for {ticker} to {csv_filename}")
        else:
            print(f"No data processed for {ticker}.")
    
    if all_historical_data_list:
        print("\nConcatenating all ticker data...")
        combined_df = pd.concat(all_historical_data_list)
        # Reset index if needed, though Date index + Ticker column should be fine for most uses
        # combined_df.reset_index(inplace=True) # Optional, consider if 'Date' column is preferred over index

        # Define the output path for the combined CSV
        # Assuming settings.PROCESSED_DATA_DIR is defined and points to a valid directory
        # If not, define a default path or ensure the directory exists.
        # For this example, saving to the current directory of the script.
        combined_csv_filename = "all_tickers_historical_features.csv"
        
        # Check if 'Date' is the index and save accordingly
        if isinstance(combined_df.index, pd.DatetimeIndex):
            combined_df.to_csv(combined_csv_filename, index=True) # Keep DatetimeIndex
        else:
            # If Date was pulled into columns by concat or reset_index without being set back
            # Or if index is just a range index after concat
            # It's generally better to have 'Date' as a column if not the primary index.
            # If 'Date' is not in columns and index is not DatetimeIndex, ensure 'Date' is preserved.
            # The original historical_data_with_tas has DatetimeIndex. pd.concat preserves it
            # if all DataFrames have the same index type (which they should).
            # If any DataFrame had a different index, concat might reset it to a RangeIndex
            # and put the original index as a column if names conflicted or were None.
            # Given fetch_yfinance_data usually returns DatetimeIndexed DFs, this should be fine.
            combined_df.to_csv(combined_csv_filename, index=False) # Or index=True if RangeIndex is desired.

        print(f"Saved all combined data to {combined_csv_filename}")
        print(f"Combined DataFrame shape: {combined_df.shape}")
        print(f"Combined DataFrame head:\n{combined_df.head()}")
        print(f"Combined DataFrame tail:\n{combined_df.tail()}")
        # Verify Ticker column presence and unique values
        if 'Ticker' in combined_df.columns:
            print(f"Unique tickers in combined data: {combined_df['Ticker'].nunique()}")
            print(f"Tickers: {combined_df['Ticker'].unique()}")
        else:
            print("Warning: 'Ticker' column not found in the combined DataFrame.")

    else:
        print("\nNo data was processed for any ticker. Combined CSV not created.")
    
    # Example: Accessing data for a specific ticker from the combined_df (if needed for quick check)
    # if not combined_df.empty and 'AAPL' in combined_df['Ticker'].unique():
    #     print("\nAAPL Features from combined DataFrame head:")
    #     print(combined_df[combined_df['Ticker'] == 'AAPL'].head())

    print("\nHistorical data fetching and TA calculation for combined output finished.")
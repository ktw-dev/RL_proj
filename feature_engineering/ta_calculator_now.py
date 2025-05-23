# feature_engineering/ta_calculator_now.py: Calculates current technical indicators and OHLCV features for real-time inference. 

import pandas as pd
# We can reuse the main calculation logic from ta_calculator
from .ta_calculator import calculate_technical_indicators, _preprocess_ohlcv # Relative import

# In a real-time scenario, you'd fetch very recent data (e.g., up to yesterday or intra-day if available)
# For this structure, we assume ohlcv_df passed to this function is already the most recent available data.

def calculate_current_technical_indicators(ohlcv_df_recent, include_ohlcv=True):
    """
    Calculates technical indicators from the most recent OHLCV data for inference.
    This function primarily wraps the historical calculator, assuming the input DataFrame
    is already appropriately filtered for the "current" timeframe (e.g., last N days needed for TA calculation).

    To get meaningful TA values for the *very last* data point (e.g., for today's inference),
    the input `ohlcv_df_recent` should contain enough prior data points to satisfy the longest
    lookback period of the TAs being calculated (e.g., 50 days for a 50-day SMA).

    :param ohlcv_df_recent: pandas DataFrame with recent OHLCV data. 
                            Should have 'Date' or 'Datetime' as index and 
                            'Open', 'High', 'Low', 'Close', 'Volume' columns.
                            It's crucial this DataFrame contains enough history for TA calculation.
    :param include_ohlcv: If True, the original OHLCV columns are included.
    :return: pandas DataFrame with current technical indicators. 
             It might return more than one row if TAs are calculated over a window, 
             the last row would typically be used for "current" inference state.
    """
    if not isinstance(ohlcv_df_recent, pd.DataFrame) or ohlcv_df_recent.empty:
        print("Input ohlcv_df_recent is empty or not a DataFrame for current TA.")
        return pd.DataFrame()

    print(f"Calculating current technical indicators on recent data (shape: {ohlcv_df_recent.shape})")
    
    # The existing calculate_technical_indicators function can be used directly.
    # It handles preprocessing, TA calculation, and NaN filling.
    current_features_df = calculate_technical_indicators(ohlcv_df_recent.copy(), include_ohlcv=include_ohlcv)

    if current_features_df.empty:
        print("Failed to calculate current technical indicators.")
        return pd.DataFrame()
    
    # The user might only want the latest row of features for inference.
    # However, this function will return all calculated rows, and the caller can decide.
    # For example, caller can do: latest_features = current_features_df.iloc[-1:]
    print(f"Successfully calculated current TAs. Result shape: {current_features_df.shape}")
    return current_features_df

if __name__ == '__main__':
    # Example Usage:
    # This example assumes you have fetched recent data (e.g., last 60 days to calculate up to 50-day TAs)
    # In a real application, this data would come from ta_fetcher.py for a specific ticker.
    
    # Create a dummy recent OHLCV DataFrame (e.g., last 60 days)
    # More data points are needed than in ta_calculator.py example to ensure TAs are meaningful for last rows
    date_rng = pd.date_range(end=pd.Timestamp.now().normalize() - pd.Timedelta(days=1), periods=60, freq='B') # Business days
    data = {
        'Date': date_rng,
        'Open': [150 + i*0.1 for i in range(60)],
        'High': [155 + i*0.1 for i in range(60)],
        'Low':  [148 + i*0.1 for i in range(60)],
        'Close':[152 + i*0.1 for i in range(60)],
        'Volume':[1000 + i*10 for i in range(60)]
    }
    sample_recent_ohlcv_df = pd.DataFrame(data)
    # sample_recent_ohlcv_df.set_index('Date', inplace=True) # _preprocess_ohlcv handles this now

    print("--- Testing TA Calculator (Current/Now) ---")
    print("Original Recent OHLCV data (sample tail):")
    print(sample_recent_ohlcv_df.tail())

    current_features = calculate_current_technical_indicators(sample_recent_ohlcv_df.copy(), include_ohlcv=True)

    if not current_features.empty:
        print("\nCalculated Current Features (last 5 rows):")
        pd.set_option('display.max_columns', None)
        print(current_features.tail())
        pd.reset_option('display.max_columns')
        
        latest_feature_set = current_features.iloc[-1:]
        print("\nLatest feature set (for inference on the last day):")
        print(latest_feature_set)
    else:
        print("\nFailed to calculate current features.")

    # Example: What if we only pass data for 1 day?
    # Most TAs will be NaN. The calculate_technical_indicators handles NaN filling,
    # but the TA values themselves would be meaningless if not enough historical data is provided.
    print("\n--- Test with insufficient historical data (e.g., only 1 day) ---")
    very_short_df = sample_recent_ohlcv_df.iloc[-1:].copy()
    short_features = calculate_current_technical_indicators(very_short_df)
    if not short_features.empty:
        print("Features for very short DF (likely many NaNs before fill or less meaningful values):")
        print(short_features)
    else:
        print("Calculation failed or returned empty for very short DF.")
    print("Note: For meaningful current TAs, ensure the input DataFrame has sufficient lookback period.") 
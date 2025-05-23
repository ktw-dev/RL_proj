# feature_engineering/ta_calculator.py: Calculates historical technical indicators and OHLCV features for TST pre-training. 

import pandas as pd
import pandas_ta as ta

# Helper function to ensure correct column names for pandas_ta
def _preprocess_ohlcv(df):
    """Prepares DataFrame for pandas_ta by renaming columns if necessary."""
    df = df.copy()
    # Ensure Date is datetime object and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    elif 'Datetime' in df.columns: # For intraday data if ever used
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df.set_index('Datetime', inplace=True)
    else:
        # Try to find a date-like index already set
        if not isinstance(df.index, pd.DatetimeIndex):
            print("Warning: No 'Date' or 'Datetime' column found and index is not DatetimeIndex. TA calculation might fail.")
            return df, False # Indicate failure to set index

    # Standardize column names (case-insensitive check)
    rename_map = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower == 'open':
            rename_map[col] = 'open'
        elif col_lower == 'high':
            rename_map[col] = 'high'
        elif col_lower == 'low':
            rename_map[col] = 'low'
        elif col_lower == 'close':
            rename_map[col] = 'close'
        elif col_lower == 'volume':
            rename_map[col] = 'volume'
        elif col_lower == 'adj close': # Keep adj close if present
            rename_map[col] = 'adj_close' 
            
    df.rename(columns=rename_map, inplace=True)
    
    # Check if essential columns are present
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Warning: DataFrame for TA calculation is missing one or more required columns: {required_cols}. Present: {df.columns.tolist()}")
        return df, False # Indicate failure due to missing columns
        
    # pandas_ta expects lowercase column names
    df.columns = df.columns.str.lower()
    return df, True

def calculate_technical_indicators(ohlcv_df, include_ohlcv=True):
    """
    Calculates a comprehensive set of technical indicators from an OHLCV DataFrame.
    """
    if not isinstance(ohlcv_df, pd.DataFrame) or ohlcv_df.empty:
        print("Input ohlcv_df is empty or not a DataFrame.")
        return pd.DataFrame()

    processed_df, success = _preprocess_ohlcv(ohlcv_df)
    if not success:
        print("Failed to preprocess OHLCV data. Skipping TA calculation.")
        return pd.DataFrame()

    # Define a very comprehensive strategy based on user's request
    # Using common default lengths where not specified, or multiple lengths for MAs
    # Updated for pandas-ta 0.3.14b compatibility
    extended_custom_strategy = ta.Strategy(
        name="ExtendedComprehensiveIndicators",
        description="A very comprehensive mix of TAs for stock prediction based on user list (pandas-ta 0.3.14b compatible)",
        ta=[
            # 1. Trend & Moving Averages
            {"kind": "sma", "length": 10, "col_names": "SMA_10"},
            {"kind": "sma", "length": 20, "col_names": "SMA_20"},
            {"kind": "sma", "length": 50, "col_names": "SMA_50"},
            {"kind": "ema", "length": 10, "col_names": "EMA_10"},
            {"kind": "ema", "length": 20, "col_names": "EMA_20"},
            {"kind": "ema", "length": 50, "col_names": "EMA_50"},
            {"kind": "wma", "length": 10, "col_names": "WMA_10"},
            {"kind": "wma", "length": 20, "col_names": "WMA_20"},
            {"kind": "dema", "length": 10, "col_names": "DEMA_10"},
            {"kind": "dema", "length": 20, "col_names": "DEMA_20"},
            {"kind": "tema", "length": 10, "col_names": "TEMA_10"},
            {"kind": "tema", "length": 20, "col_names": "TEMA_20"},
            {"kind": "trima", "length": 10, "col_names": "TRIMA_10"},
            {"kind": "trima", "length": 20, "col_names": "TRIMA_20"},
            {"kind": "kama", "length": 10, "col_names": "KAMA_10"}, 
            {"kind": "kama", "length": 20, "col_names": "KAMA_20"},
            # {"kind": "mama", "fast": 0.5, "slow": 0.05, "col_names": ("MAMA", "FAMA")}, # MAMA is not directly available or requires specific parameters not easily defaulted in Strategy
            {"kind": "t3", "length": 10, "col_names": "T3_10"},
            {"kind": "vwap"}, # VWAP in 0.3.14b typically doesn't require col_names if defaults are used, it creates VWAP_D

            # 2. Momentum
            {"kind": "macd", "col_names": ("MACD_12_26_9", "MACDh_12_26_9", "MACDs_12_26_9")}, # Adjusted col_names for clarity
            {"kind": "mom", "length": 10, "col_names": "MOM_10"},
            {"kind": "mom", "length": 20, "col_names": "MOM_20"},
            {"kind": "rsi", "length": 14, "col_names": "RSI_14"},
            {"kind": "rsi", "length": 21, "col_names": "RSI_21"},
            {"kind": "stoch", "k": 14, "d": 3, "smooth_k": 3, "col_names": ("STOCHk_14_3_3", "STOCHd_14_3_3")}, 
            {"kind": "stoch", "k": 5, "d": 3, "smooth_k": 1, "col_names": ("STOCHFk_5_3_1", "STOCHFd_5_3_1")}, 
            {"kind": "stochrsi", "length": 14, "rsi_length": 14, "k": 3, "d": 3, "col_names": ("STOCHRSIk_14_14_3_3", "STOCHRSId_14_14_3_3")},
            {"kind": "willr", "length": 14, "col_names": "WILLR_14"},
            {"kind": "cci", "length": 14, "col_names": "CCI_14"},
            {"kind": "cci", "length": 20, "col_names": "CCI_20"},
            {"kind": "cmo", "length": 14, "col_names": "CMO_14"},
            {"kind": "roc", "length": 10, "col_names": "ROC_10"},
            {"kind": "roc", "length": 20, "col_names": "ROC_20"},
            {"kind": "roc", "length": 10, "percent": False, "col_names": "ROCR_10"}, 
            {"kind": "apo", "fast": 12, "slow": 26, "col_names": "APO_12_26"}, 
            {"kind": "ppo", "fast": 12, "slow": 26, "signal": 9, "col_names": ("PPO_12_26_9", "PPOh_12_26_9", "PPOs_12_26_9")}, 
            {"kind": "trix", "length": 14, "col_names": "TRIX_14"},
            {"kind": "uo", "fast": 7, "medium": 14, "slow": 28, "col_names": "UO_7_14_28"}, # ultosc is 'uo' in pandas-ta

            # 3. Trend Strength / Directional
            {"kind": "adx", "length": 14, "col_names": ("ADX_14", "DMP_14", "DMN_14")}, 
            # {"kind": "adxr", "length": 14, "col_names": "ADXR_14"}, # adxr might require adx to be calculated first or is part of vortex
            {"kind": "aroon", "length": 14, "col_names": ("AROOND_14", "AROONU_14", "AROONOSC_14")},
            # {"kind": "dx", "length": 14, "col_names": "DX_14"}, # dx is often part of adx calculation
            {"kind": "dm", "length": 14, "col_names": ("PDM_14", "MDM_14")}, 
            
            # 4. Volatility & Bands
            {"kind": "bbands", "length": 20, "std": 2, "col_names": ("BBL_20_2", "BBM_20_2", "BBU_20_2", "BBB_20_2", "BBP_20_2")},
            {"kind": "atr", "length": 14, "col_names": "ATR_14"},
            {"kind": "natr", "length": 14, "col_names": "NATR_14"},
            {"kind": "true_range", "col_names": "TRUERANGE"},
            {"kind": "psar", "col_names": ("PSARl", "PSARs")}, 
            {"kind": "midpoint", "length": 14, "col_names": "MIDPOINT_14"},
            {"kind": "midprice", "length": 14, "col_names": "MIDPRICE_14"},
            {"kind": "kc", "length": 20, "atr_length": 10, "col_names": ("KCUe_20_10", "KCLe_20_10", "KCM_20_10")}, 

            # 5. Volume & Flow
            {"kind": "obv", "col_names": "OBV"},
            {"kind": "ad", "col_names": "AD"}, 
            {"kind": "adosc", "fast": 3, "slow": 10, "col_names": "ADOSC_3_10"}, 
            {"kind": "mfi", "length": 14, "col_names": "MFI_14"},
            {"kind": "bop", "col_names": "BOP"}, 
            # VWAP already included in Trend/MA section

            # 6. Hilbert Transform Series (ht() provides multiple outputs)
            # {"kind": "ht", "col_names": ("HT_DCPERIOD", "HT_DCPHASE", "HT_PHASOR_I", "HT_PHASOR_Q", "HT_SINE", "HT_LEADSINE", "HT_TRENDMODE")} # ht provides multiple columns, col_names might need adjustment or let default
            {"kind": "cdl_z"} # Example of a candle pattern, many are available
        ]
    )

    try:
        print(f"Calculating extended TAs for DataFrame with index: {processed_df.index.name}, columns: {processed_df.columns.tolist()}")
        processed_df.ta.strategy(extended_custom_strategy)
    except Exception as e:
        print(f"Error calculating extended technical indicators: {e}")
        print(f"DataFrame columns at error time: {processed_df.columns.tolist()}")
        print(f"DataFrame head:\\n{processed_df.head()}")
        # Attempt to see which part of the strategy might have failed if possible
        # for item in extended_custom_strategy.ta:
        #     try:
        #         print(f"Attempting: {item['kind']}")
        #         if 'col_names' in item and isinstance(item['col_names'], str) and ',' in item['col_names']:
        #             item_strategy = ta.Strategy(name="test_item", ta=[{**item, 'col_names': tuple(c.strip() for c in item['col_names'].split(','))}])
        #         else:
        #             item_strategy = ta.Strategy(name="test_item", ta=[item])
        #         print(f"Strategy for item: {item_strategy.ta}")
        #         current_cols = processed_df.columns.tolist()
        #         processed_df.ta.strategy(item_strategy)
        #         new_cols = [col for col in processed_df.columns if col not in current_cols]
        #         if not new_cols: print(f"No new columns for {item['kind']}")
        #         else: print(f"Indicator {item['kind']} added columns: {new_cols}")
        #     except Exception as item_e:
        #         print(f"Failed on indicator: {item['kind']} with error: {item_e}")
        #         # break # Stop on first error to isolate
        return pd.DataFrame()

    if not include_ohlcv:
        cols_to_drop = ['open', 'high', 'low', 'close', 'volume']
        if 'adj_close' in processed_df.columns and 'adj_close' not in cols_to_drop: pass # Don't drop adj_close if present and asked to keep OHLCV conceptually
        processed_df.drop(columns=[col for col in cols_to_drop if col in processed_df.columns], inplace=True)
    
    processed_df.ffill(inplace=True)
    processed_df.bfill(inplace=True)
    
    return processed_df

if __name__ == '__main__':
    # Example Usage:
    # Create a dummy OHLCV DataFrame that mimics the raw data structure
    # (e.g., as obtained from fetch_yfinance_data before TA calculation within ta_fetcher_history.py)
    # This raw form typically has string Dates and capitalized column names.
    
    num_periods = 70  # Increased for better TA calculation, e.g., SMA_50 needs ~50 periods
    
    # Generate business dates and format them as strings
    # Using a fixed start date for reproducibility of the example
    try:
        base_date = pd.to_datetime('2023-01-01')
        dates_as_strings = pd.date_range(
            start=base_date,
            periods=num_periods,
            freq='B'  # Business days
        ).strftime('%Y-%m-%d').tolist()
    except Exception as e:
        # Fallback if date generation has issues (e.g. pandas version differences)
        print(f"Date generation error: {e}, creating simple sequential dates.")
        dates_as_strings = [f"2023-{((i//30)+1):02d}-{((i%30)+1):02d}" for i in range(num_periods)]

    # Ensure the dates are sorted, as they would be from historical fetching
    # dates_as_strings.sort() # date_range already returns sorted dates

    data = {
        'Date': dates_as_strings,
        'Open':   [150 + i*0.1 + (i%5)*0.5 - (i%3)*0.3 for i in range(num_periods)],
        'High':   [155 + i*0.2 + (i%4)*0.6 - (i%2)*0.2 for i in range(num_periods)],
        'Low':    [148 + i*0.05 - (i%5)*0.4 + (i%3)*0.2 for i in range(num_periods)], # Low should generally be lower than open/close
        'Close':  [152 + i*0.08 + (i%3)*0.4 - (i%4)*0.3 for i in range(num_periods)],
        'Volume': [100000 + i*1000 + (i%10)*500 - (i%7)*300 for i in range(num_periods)],
        'Adj Close':[151.5 + i*0.08 + (i%3)*0.4 - (i%4)*0.3 for i in range(num_periods)] # Kept similar to Close for simplicity
    }
    # Ensure Low is not greater than High, Open, or Close, and High is not less than Low, Open, Close
    for i in range(num_periods):
        data['Low'][i] = min(data['Low'][i], data['Open'][i], data['Close'][i], data['High'][i] - 0.01) # Ensure Low <= others
        data['High'][i] = max(data['High'][i], data['Open'][i], data['Close'][i], data['Low'][i] + 0.01)   # Ensure High >= others
        if data['Open'][i] > data['High'][i]: data['Open'][i] = data['High'][i]
        if data['Open'][i] < data['Low'][i]: data['Open'][i] = data['Low'][i]
        if data['Close'][i] > data['High'][i]: data['Close'][i] = data['High'][i]
        if data['Close'][i] < data['Low'][i]: data['Close'][i] = data['Low'][i]
        if data['Volume'][i] < 0: data['Volume'][i] = 0 # Ensure volume is not negative
        
    sample_ohlcv_df_raw = pd.DataFrame(data)

    print("--- Testing Extended TA Calculator (Historical) with raw-like data ---")
    print("Sample raw OHLCV data (mimicking fetch_yfinance_data output, head):")
    print(sample_ohlcv_df_raw.head())

    # This is the core call, demonstrating the calculation process on raw-like data
    features_df = calculate_technical_indicators(sample_ohlcv_df_raw.copy(), include_ohlcv=True)

    if not features_df.empty:
        print("\nCalculated Features (sample head):")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', 10) # Show a few more rows
        print(features_df.head(10))
        print("\nCalculated Features (sample tail):")
        print(features_df.tail(10))
        pd.reset_option('display.max_rows')
        pd.reset_option('display.max_columns')
        print(f"\nTotal number of features calculated (including OHLCVA if kept): {len(features_df.columns)}")
        print(f"Shape of features_df: {features_df.shape}")
        
        nan_counts = features_df.isnull().sum()
        print(f"Total NaNs remaining after ffill/bfill: {nan_counts.sum()}")
        if nan_counts.sum() > 0:
            print("Columns with NaNs after fill:")
            print(nan_counts[nan_counts > 0])
    else:
        print("\nFailed to calculate extended features.")

    print("\n--- Test with missing essential column ---")
    # Create a new DataFrame for this test to avoid modifying sample_ohlcv_df_raw
    data_for_missing_test = sample_ohlcv_df_raw.copy()
    missing_col_df = data_for_missing_test.drop(columns=['Volume'])
    failed_features = calculate_technical_indicators(missing_col_df.copy())
    if failed_features.empty:
        print("Correctly returned empty DataFrame for missing essential columns.")
    else:
        print("Error: Calculation proceeded despite missing columns.") 
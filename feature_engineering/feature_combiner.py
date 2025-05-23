# feature_engineering/feature_combiner.py: Combines technical analysis features and news sentiment features.

import pandas as pd
import numpy as np

def align_and_combine_features(
    ta_features_df: pd.DataFrame, 
    news_features_df: pd.DataFrame, 
    default_news_fill_value: float = 0.0
) -> pd.DataFrame:
    """
    Combines historical technical analysis (TA) features with news sentiment features for model training.

    It aligns the two DataFrames based on a (Date, Ticker) multi-index and performs a left join.
    News features are forward-filled within each ticker group after alignment.

    Args:
        ta_features_df (pd.DataFrame): DataFrame with technical indicators. Expected to have a 
                                       DatetimeIndex (e.g., 'Date') and a 'Ticker' column, or 
                                       a (Date, Ticker) MultiIndex.
        news_features_df (pd.DataFrame): DataFrame with processed news sentiment features. Expected 
                                         to have a similar structure: DatetimeIndex (e.g., 'Date') 
                                         and a 'Ticker' column, or a (Date, Ticker) MultiIndex.
                                         For "neutral news" training, this df should contain the 
                                         7 sentiment columns populated with neutral values.
        default_news_fill_value (float, optional): Value to fill NaN news features after alignment 
                                                   and forward-fill. Defaults to 0.0.

    Returns:
        pd.DataFrame: Combined features, with a (Date, Ticker) MultiIndex, ready for TST model input.
                      Returns an empty DataFrame on critical error.
    """
    if not isinstance(ta_features_df, pd.DataFrame) or ta_features_df.empty:
        print("TA features DataFrame is empty or invalid. Cannot combine.")
        return pd.DataFrame()
        
    if not isinstance(news_features_df, pd.DataFrame): # news_features_df can be empty if no news
        print("News features DataFrame is invalid (not a DataFrame object). Joining TA with no news features.")
        # If news is optional and not provided, we might just return TA features or TA + placeholder neutral news cols.
        # For this version, we assume news_features_df is provided, even if it's to represent neutral state.
        # If it's truly absent (None or invalid type), we might add default neutral columns to ta_df.
        # For now, if it's an invalid type, error out. If it's an empty DF, specific handling below.
        return pd.DataFrame() 

    ta_df = ta_features_df.copy()
    news_df = news_features_df.copy()

    print("Combining historical TA and News features for training...")

    # --- Standardize TA DataFrame --- 
    # Ensure TA features have 'Date' as index and 'Ticker' as column, then create (Date, Ticker) MultiIndex
    if not isinstance(ta_df.index, pd.MultiIndex) or not (isinstance(ta_df.index.names, list) and 'Date' in ta_df.index.names and 'Ticker' in ta_df.index.names):
        if not isinstance(ta_df.index, pd.DatetimeIndex):
            if 'Date' in ta_df.columns:
                ta_df['Date'] = pd.to_datetime(ta_df['Date'])
                ta_df.set_index('Date', inplace=True)
            else:
                print("Error: TA features must have a DatetimeIndex or a 'Date' column.")
                return pd.DataFrame()
        
        if 'Ticker' in ta_df.columns:
            ta_df.set_index(['Ticker'], append=True, inplace=True) # Now (Date, Ticker) MI
        else:
            print("Error: TA features must have a 'Ticker' column if not already in a (Date, Ticker) MultiIndex.")
            return pd.DataFrame()
    # Ensure MI names are standardized if they came in as MI
    ta_df.index.names = ['Date', 'Ticker']

    # --- Standardize News DataFrame --- 
    # Ensure News features have 'Date' as index and 'Ticker' as column, then create (Date, Ticker) MultiIndex
    if not news_df.empty:
        if not isinstance(news_df.index, pd.MultiIndex) or not (isinstance(news_df.index.names, list) and 'Date' in news_df.index.names and 'Ticker' in news_df.index.names):
            # Handle index name if it's 'date' (from news_processor) or 'published_date'
            current_date_col_name = None
            if isinstance(news_df.index, pd.DatetimeIndex):
                if news_df.index.name is not None and news_df.index.name != 'Date': # e.g. 'date'
                    current_date_col_name = news_df.index.name
                    news_df.index.name = 'Date' 
            elif 'Date' in news_df.columns:
                news_df['Date'] = pd.to_datetime(news_df['Date'])
                news_df.set_index('Date', inplace=True)
            elif 'published_date' in news_df.columns: # Legacy or alternative name
                news_df['published_date'] = pd.to_datetime(news_df['published_date'])
                news_df.rename(columns={'published_date': 'Date'}, inplace=True)
                news_df.set_index('Date', inplace=True)
            elif 'date' in news_df.columns: # if 'date' (from news_processor) was a column
                news_df['date'] = pd.to_datetime(news_df['date'])
                news_df.rename(columns={'date': 'Date'}, inplace=True)
                news_df.set_index('Date', inplace=True)
            else:
                print("Error: News features must have a DatetimeIndex (named 'Date', 'date', or 'published_date') or a corresponding column.")
                return pd.DataFrame()
            
            if 'Ticker' in news_df.columns:
                news_df.set_index(['Ticker'], append=True, inplace=True) # Now (Date, Ticker) MI
            elif not isinstance(news_df.index, pd.MultiIndex): # If it became MI some other way, or error
                print("Error: News features must have a 'Ticker' column if not already in a (Date, Ticker) MultiIndex.")
                return pd.DataFrame()
        # Ensure MI names are standardized
        news_df.index.names = ['Date', 'Ticker']
    
    # --- Combine DataFrames --- 
    if not news_df.empty:
        # Reindex news_df to the (Date, Ticker) multi-index of ta_df.
        # This aligns news data to TA dates, introducing NaNs where news is missing on a TA date.
        aligned_news_df = news_df.reindex(ta_df.index)
        
        # Forward fill news data *within each ticker group*.
        # This propagates the last known news sentiment until a new one appears.
        if not aligned_news_df.empty:
             aligned_news_df = aligned_news_df.groupby(level='Ticker').ffill()
        
        # Fill any initial NaNs in news (e.g., if a stock has no news before its first TA date, 
        # or if the synthetic neutral news_df was somehow incomplete for certain early dates).
        if default_news_fill_value is not None and not aligned_news_df.empty:
            # Define specific fill values for neutral sentiment columns if default_news_fill_value is too generic
            # For a general combiner, default_news_fill_value (e.g. 0.0) is applied to all news NaNs.
            # If this function KNEW it was for neutral news that should be 0,0,1,0,0,0,0, then a more complex fill could be here.
            # However, it's better if the input news_df is already perfectly neutral if that's the goal.
            # This fill is a fallback.
            aligned_news_df.fillna(default_news_fill_value, inplace=True)

        # Join TA with aligned news features. Left join keeps all TA entries.
        combined_df = ta_df.join(aligned_news_df, how='left')
    else: # news_df is empty, so only TA features are used.
          # Optionally, add placeholder columns for news features with neutral values if model expects them.
        print("News features DataFrame is empty. Proceeding with TA features only.")
        print("If the model expects news columns, they should be added here with default neutral values.")
        combined_df = ta_df.copy()
        # Example: Adding expected neutral news columns if news_df was empty
        expected_news_cols = [
            'avg_sentiment_positive', 'avg_sentiment_negative', 'avg_sentiment_neutral',
            'news_count', 'weekend_effect_positive', 'weekend_effect_negative', 'weekend_effect_neutral'
        ]
        # This part is crucial: what are the truly neutral values?
        # avg_neutral = 1.0, others = 0.0 for sentiment. news_count = 0.
        if not all(col in combined_df.columns for col in expected_news_cols):
            print(f"Adding placeholder neutral news columns with default_fill_value: {default_news_fill_value} (and 1.0 for avg_sentiment_neutral)")
            combined_df['avg_sentiment_positive'] = default_news_fill_value
            combined_df['avg_sentiment_negative'] = default_news_fill_value
            combined_df['avg_sentiment_neutral'] = 1.0 # Key neutral value
            combined_df['news_count'] = default_news_fill_value # Typically 0 for no news
            combined_df['weekend_effect_positive'] = default_news_fill_value
            combined_df['weekend_effect_negative'] = default_news_fill_value
            combined_df['weekend_effect_neutral'] = default_news_fill_value # Or 1.0 if neutral effect propagates
            # Ensure new columns are float type
            for col in expected_news_cols:
                 if col in combined_df.columns:
                    combined_df[col] = combined_df[col].astype(float)

    # Final NaN handling for the entire combined DataFrame (especially for TA features from early in series)
    # Fill NaNs by first forward-filling, then backward-filling within each ticker group.
    if isinstance(combined_df.index, pd.MultiIndex) and 'Ticker' in combined_df.index.names:
         combined_df = combined_df.groupby(level='Ticker').ffill().bfill()
    else: # Should ideally be a MultiIndex by now
         combined_df.ffill(inplace=True)
         combined_df.bfill(inplace=True)
    
    # Drop rows if all features are NaN after fill (can happen if a ticker has very short history)
    combined_df.dropna(how='all', inplace=True)

    print(f"Combined features ready. Shape: {combined_df.shape}")
    if not combined_df.empty:
        print(f"Columns in combined_df: {combined_df.columns.tolist()}")
        # print(f"NaN count per column:\n{combined_df.isnull().sum()}")
    return combined_df

if __name__ == '__main__':
    print("\n--- Testing Feature Combiner (Training Data Scenario) ---")
    # Dummy TA Data (Index: Date, Column: Ticker)
    dates_hist = pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-02', '2023-01-02', '2023-01-03', '2023-01-03', '2023-01-04'])
    tickers_hist = ['AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL', 'MSFT', 'AAPL']
    sample_ta_hist_df = pd.DataFrame({
        'Date': dates_hist,
        'Ticker': tickers_hist,
        'SMA_10': [150, 290, 151, 291, 152, 292, np.nan], # AAPL has a NaN for SMA_10 on last day
        'RSI_14': [50, 55, 52, 56, 53, 57, 54]
    })
    sample_ta_hist_df.set_index('Date', inplace=True)

    # Dummy News Data (Synthetically Neutral, Index: Date, Column: Ticker)
    # Should cover all dates/tickers in TA data for perfect alignment.
    # Here, let's make it sparse to test ffill and default_news_fill_value
    news_dates_hist = pd.to_datetime(['2023-01-01', '2023-01-01', '2023-01-03', '2023-01-03'])
    news_tickers_hist = ['AAPL', 'MSFT', 'AAPL', 'MSFT']
    sample_news_hist_df = pd.DataFrame({
        'Date': news_dates_hist,
        'Ticker': news_tickers_hist,
        'avg_sentiment_positive': [0.0, 0.0, 0.0, 0.0],
        'avg_sentiment_negative': [0.0, 0.0, 0.0, 0.0],
        'avg_sentiment_neutral': [1.0, 1.0, 1.0, 1.0],
        'news_count': [1, 1, 1, 1],
        'weekend_effect_positive': [0.0, 0.0, 0.0, 0.0],
        'weekend_effect_negative': [0.0, 0.0, 0.0, 0.0],
        'weekend_effect_neutral': [0.0, 0.0, 0.0, 0.0]
    })
    sample_news_hist_df.set_index('Date', inplace=True)

    print("\nSample Historical TA (before setting MI):")
    print(sample_ta_hist_df)
    print("\nSample Historical News Features (before setting MI):")
    print(sample_news_hist_df)

    combined_hist_df = align_and_combine_features(sample_ta_hist_df.copy(), sample_news_hist_df.copy(), default_news_fill_value=0.0)
    
    print("\nCombined Historical Features (Index should be Date, Ticker MultiIndex):")
    if not combined_hist_df.empty:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 2000)
        print(combined_hist_df)
        print(f"Shape: {combined_hist_df.shape}")
        print(f"Index: {combined_hist_df.index.names}")
        # Check NaNs. SMA_10 for AAPL on 01-04 was NaN, should be filled by bfill from 01-03 value.
        # News for 01-02 and 01-04 (AAPL) should be filled by ffill from 01-01/01-03 then default or bfill.
        print(f"NaNs remaining after all fills: {combined_hist_df.isnull().sum().sum()}")
        # Expected NaNs = 0 if all Tickers had some data to bfill from.
        # Specifically, AAPL on 2023-01-04 had NaN SMA_10. It should be filled by bfill from 152.
        # News for AAPL on 2023-01-02 should be ffilled from 2023-01-01.
        # News for AAPL on 2023-01-04 should be ffilled from 2023-01-03.
        if not combined_hist_df.empty and 'AAPL' in combined_hist_df.index.get_level_values('Ticker'):
            print("\nAAPL Data from combined:")
            print(combined_hist_df.xs('AAPL', level='Ticker'))
            assert combined_hist_df.xs('AAPL', level='Ticker').loc[pd.to_datetime('2023-01-04')]['SMA_10'] == 152.0, "SMA_10 bfill failed"
            assert combined_hist_df.xs('AAPL', level='Ticker').loc[pd.to_datetime('2023-01-02')]['avg_sentiment_neutral'] == 1.0, "News ffill failed for 01-02"
            assert combined_hist_df.xs('AAPL', level='Ticker').loc[pd.to_datetime('2023-01-04')]['avg_sentiment_neutral'] == 1.0, "News ffill failed for 01-04"

    else:
        print("Historical combination failed.")

    print("\n--- Test with empty news features DataFrame ---")
    combined_no_news = align_and_combine_features(sample_ta_hist_df.copy(), pd.DataFrame(), default_news_fill_value=0.0)
    print("\nCombined Features (Empty News Provided):")
    if not combined_no_news.empty:
        print(combined_no_news)
        print(f"Shape: {combined_no_news.shape}")
        # Check if neutral news columns were added
        expected_news_cols = ['avg_sentiment_neutral']
        assert all(col in combined_no_news.columns for col in expected_news_cols), "Neutral news columns not added for empty news input."
        assert combined_no_news.xs('AAPL', level='Ticker').loc[pd.to_datetime('2023-01-01')]['avg_sentiment_neutral'] == 1.0

    else:
        print("Combination with empty news failed.") 
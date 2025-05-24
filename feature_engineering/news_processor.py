# feature_engineering/news_processor.py: Processes analyzed news data for feature engineering.
# 05-24-2025 19:28

import pandas as pd
import numpy as np
from datetime import timedelta, date

# Helper to identify US business days (simplified, assuming no specific holiday calendar needed for now)
def is_us_business_day(dt_date: date):
    # Monday to Friday
    return dt_date.weekday() < 5

def get_next_n_business_days(start_date: date, n: int):
    """Returns a list of the next N business days starting from start_date (inclusive if business day)."""
    business_days = []
    current_date = start_date
    while len(business_days) < n:
        if is_us_business_day(current_date):
            business_days.append(current_date)
        current_date += timedelta(days=1)
    return business_days

def aggregate_daily_sentiment_features(analyzed_news_df: pd.DataFrame, ticker_symbol: str):
    """
    Aggregates per-headline sentiment scores from news_analyzer.py into daily sentiment features.
    Handles weekend news by applying its sentiment to the following week's business days.

    Args:
        analyzed_news_df (pd.DataFrame): DataFrame from news_analyzer.py. 
                                         Expected columns: ['ticker', 'published_date', 'published_datetime_utc',
                                                         'headline', 'sentiment_positive', 
                                                         'sentiment_negative', 'sentiment_neutral'].
                                         'published_date' should be a datetime.date object.
        ticker_symbol (str): The ticker for which news is being processed (for clarity in logs/output).

    Returns:
        pd.DataFrame: Columns: ['date', 'avg_sentiment_positive', 'avg_sentiment_negative', 
                                'avg_sentiment_neutral', 'news_count', 'weekend_effect_positive',
                                'weekend_effect_negative', 'weekend_effect_neutral'], indexed by 'date'.
                      Contains features for each relevant business day based on the input news window.
    """
    if not isinstance(analyzed_news_df, pd.DataFrame) or analyzed_news_df.empty:
        print(f"No analyzed news data provided for {ticker_symbol}. Returning empty DataFrame.")
        return pd.DataFrame()

    df = analyzed_news_df.copy()

    required_cols = ['published_date', 'sentiment_positive', 'sentiment_negative', 'sentiment_neutral']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Analyzed news DataFrame for {ticker_symbol} is missing required columns: {required_cols}. Has: {df.columns.tolist()}")
        return pd.DataFrame()
    
    # Ensure 'published_date' is indeed a date object, not datetime, for grouping
    if not df.empty and isinstance(df['published_date'].iloc[0], pd.Timestamp):
         df['published_date'] = pd.to_datetime(df['published_date']).dt.date
    elif not df.empty and isinstance(df['published_date'].iloc[0], str):
         df['published_date'] = pd.to_datetime(df['published_date']).dt.date

    # 1. Aggregate sentiment for each actual news publication day
    daily_aggregated_sentiments = df.groupby('published_date').agg(
        avg_sentiment_positive=('sentiment_positive', 'mean'),
        avg_sentiment_negative=('sentiment_negative', 'mean'),
        avg_sentiment_neutral=('sentiment_neutral', 'mean'),
        news_count=('headline', 'count')
    ).reset_index()
    daily_aggregated_sentiments.rename(columns={'published_date': 'date'}, inplace=True)

    # Initialize weekend effect columns
    for col_suffix in ['positive', 'negative', 'neutral']:
        daily_aggregated_sentiments[f'weekend_effect_{col_suffix}'] = 0.0 # Default to neutral effect

    # 2. Identify weekend news and propagate its effect
    # A weekend is Saturday (5) or Sunday (6)
    weekend_news_list = []
    for _, row in daily_aggregated_sentiments.iterrows():
        current_date = row['date']
        if not is_us_business_day(current_date): # It's a weekend day with news
            weekend_sentiment = {
                'positive': row['avg_sentiment_positive'],
                'negative': row['avg_sentiment_negative'],
                'neutral': row['avg_sentiment_neutral']
            }
            # Find the next 5 business days starting from the Monday after this weekend news
            # If news is on Sat, next Mon. If news is on Sun, next Mon.
            start_propagation_date = current_date + timedelta(days=(7 - current_date.weekday())) # Next Monday
            target_business_days = get_next_n_business_days(start_propagation_date, 5)
            
            for biz_day in target_business_days:
                weekend_news_list.append({
                    'date': biz_day,
                    'weekend_effect_positive': weekend_sentiment['positive'],
                    'weekend_effect_negative': weekend_sentiment['negative'],
                    'weekend_effect_neutral': weekend_sentiment['neutral']
                })
    
    if weekend_news_list:
        weekend_effects_df = pd.DataFrame(weekend_news_list)
        # If multiple weekend news affect the same future business day, average their effects
        weekend_effects_df = weekend_effects_df.groupby('date').agg({
            'weekend_effect_positive': 'mean',
            'weekend_effect_negative': 'mean',
            'weekend_effect_neutral': 'mean'
        }).reset_index()

        # Merge these weekend effects into the main daily_aggregated_sentiments table
        # We need to ensure all relevant dates are present for merging.
        # Create a union of dates from daily_aggregated_sentiments and weekend_effects_df
        all_dates = pd.concat([
            daily_aggregated_sentiments[['date']], 
            weekend_effects_df[['date']]
        ]).drop_duplicates()['date'].sort_values()
        
        # Merge daily aggregates
        final_df = pd.merge(all_dates.to_frame(), daily_aggregated_sentiments, on='date', how='left')
        # Merge weekend effects
        final_df = pd.merge(final_df, weekend_effects_df, on='date', how='left')

        # Fill NaNs for weekend_effect columns that were not affected with 0 (neutral)
        for col_suffix in ['positive', 'negative', 'neutral']:
            final_df[f'weekend_effect_{col_suffix}'].fillna(0.0, inplace=True)
            # If a business day had no news itself but has a weekend effect, its avg_sentiment will be NaN.
            # We can choose to fill these NaNs if needed, e.g., with 0 or a neutral value if a day has ONLY weekend effect.
            # For now, avg_sentiments remain NaN if no direct news on that day.
            final_df[f'avg_sentiment_{col_suffix}'].fillna(0.0, inplace=True) # Assuming 0 if no news
            final_df['news_count'].fillna(0, inplace=True)
    else: # No weekend news to process
        final_df = daily_aggregated_sentiments

    # Ensure final_df only contains business days for the output, as weekend sentiments are now effects.
    # However, main.py will likely align this with TA data which is on business days.
    # For now, let final_df contain all dates for which either direct news or weekend effect exists.
    final_df.set_index('date', inplace=True)
    final_df.sort_index(inplace=True)

    # The output should cover the relevant 7-day business window for main.py
    # This processor might generate data for a slightly wider range due to weekend propagation.
    # Filtering to the exact 7 business day window will be handled by the calling function in main.py
    # or by the feature combiner when aligning with TA data.

    print(f"Processed daily sentiment features for {ticker_symbol}. Shape: {final_df.shape}")
    return final_df

if __name__ == '__main__':
    print("--- Testing News Processor ---_news_processor")
    # Create a dummy analyzed_news_df (output of news_analyzer.py)
    sample_data = []
    base_date = date(2023, 10, 20) # A Friday

    # Business day news
    for i in range(5): # Mon-Fri of the week after
        d = base_date + timedelta(days=3+i) # Mon (10/23) to Fri (10/27)
        if is_us_business_day(d):
            for j in range(3): # 3 news items per day
                sample_data.append({
                    'ticker': 'AAPL', 'published_date': d, 'published_datetime_utc': pd.Timestamp(d, tz='UTC'),
                    'headline': f'News {j} on {d}', 
                    'sentiment_positive': 0.2 + (i*0.05) + (j*0.01), 
                    'sentiment_negative': 0.1 - (i*0.01), 
                    'sentiment_neutral': 0.7 - (i*0.04) - (j*0.01)
                })

    # Weekend news (Sat 10/21, Sun 10/22)
    sat_date = base_date + timedelta(days=1) # Sat 10/21
    sun_date = base_date + timedelta(days=2) # Sun 10/22
    for j in range(2):
        sample_data.append({
            'ticker': 'AAPL', 'published_date': sat_date, 'published_datetime_utc': pd.Timestamp(sat_date, tz='UTC'),
            'headline': f'Weekend Sat News {j}', 
            'sentiment_positive': 0.6, 'sentiment_negative': 0.1, 'sentiment_neutral': 0.3
        })
        sample_data.append({
            'ticker': 'AAPL', 'published_date': sun_date, 'published_datetime_utc': pd.Timestamp(sun_date, tz='UTC'),
            'headline': f'Weekend Sun News {j}', 
            'sentiment_positive': 0.1, 'sentiment_negative': 0.7, 'sentiment_neutral': 0.2
        })

    analyzed_df_sample = pd.DataFrame(sample_data)
    # Ensure published_date is date object
    analyzed_df_sample['published_date'] = pd.to_datetime(analyzed_df_sample['published_date']).dt.date

    print("Input analyzed_news_df sample (first 5 rows):")
    print(analyzed_df_sample.head())

    processed_sentiment_features = aggregate_daily_sentiment_features(analyzed_df_sample, ticker_symbol="AAPL")

    if not processed_sentiment_features.empty:
        print("\nProcessed Daily Sentiment Features:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(processed_sentiment_features)
        # Expected: 
        # - Dates for Mon-Fri (10/23-10/27) should have avg_sentiments from their own news.
        # - These Mon-Fri dates should also have weekend_effect scores derived from Sat (10/21) and Sun (10/22) news, averaged.
        # - Weekend dates (10/21, 10/22) might appear with their direct news aggregates but weekend_effect=0 for themselves.
        #   Or we might filter them out if only business days are needed in final output.
        #   The current implementation might list them if they had direct news.

        # Let's verify weekend effect propagation for Monday 10/23
        if date(2023,10,23) in processed_sentiment_features.index:
            mon_data = processed_sentiment_features.loc[date(2023,10,23)]
            print(f"\nData for Monday 2023-10-23 (should have weekend effect):")
            print(f"{mon_data}")
            # Expected weekend effect for 10/23: avg of (0.6 pos, 0.1 neg from Sat) and (0.1 pos, 0.7 neg from Sun)
            # = (0.35 pos, 0.4 neg)
            assert abs(mon_data['weekend_effect_positive'] - 0.35) < 0.01
            assert abs(mon_data['weekend_effect_negative'] - 0.40) < 0.01
        else:
            print("Monday 2023-10-23 not found in processed features index.")

    else:
        print("\nFailed to process news features.") 
#!/usr/bin/env python3
"""
Simple script to test CSV data structure understanding
"""

def test_csv_structure():
    import csv
    import os
    
    csv_file = "all_tickers_historical_features.csv"
    
    if not os.path.exists(csv_file):
        print(f"CSV file {csv_file} not found")
        return
    
    print(f"Testing CSV structure: {csv_file}")
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        
        # Read header
        header = next(reader)
        print(f"Number of columns: {len(header)}")
        print(f"First 10 columns: {header[:10]}")
        print(f"Last 10 columns: {header[-10:]}")
        
        # Find Ticker column
        ticker_col_idx = None
        for i, col in enumerate(header):
            if 'Ticker' in col:
                ticker_col_idx = i
                break
        
        if ticker_col_idx is not None:
            print(f"Ticker column found at index: {ticker_col_idx}")
        else:
            print("Ticker column not found!")
            return
        
        # Read data rows and track tickers
        tickers_found = set()
        sample_rows = []
        
        for i, row in enumerate(reader):
            if len(row) > ticker_col_idx:
                ticker = row[ticker_col_idx]
                tickers_found.add(ticker)
                
                if i < 10:  # First 10 rows
                    sample_rows.append((row[0], ticker))
                
                # Show some rows from each ticker transition
                if len(sample_rows) < 20 or i % 5000 == 0:
                    sample_rows.append((row[0], ticker))
        
        print(f"\nUnique tickers found: {sorted(list(tickers_found))}")
        print(f"Total number of tickers: {len(tickers_found)}")
        
        print(f"\nSample rows (Date, Ticker):")
        for date, ticker in sample_rows[:20]:
            print(f"  {date} -> {ticker}")
    
    print("\nStructure test completed!")

if __name__ == "__main__":
    test_csv_structure() 
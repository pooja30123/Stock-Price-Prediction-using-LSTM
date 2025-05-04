from datetime import datetime, date, timedelta
import yfinance as yf
import os
import pandas as pd
import os,sys
sys.path.append(os.path.abspath('..'))

def download_stock_data(symbol: str):
    """
    Download stock data from Yahoo Finance from Jan 1, 2025 to today.
    """
    try:
        # Get today's date and the date two years ago
        end_date = datetime.today().strftime('%Y-%m-%d')
        start_date = '2025-01-01'

        # Ensure the 'data' directory exists, create if not
        directory = 'data'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Download the stock data
        stock_data = yf.download(symbol, start=start_date, end=end_date, auto_adjust=False)
        
        # Save the downloaded data to a CSV file in the 'data' directory
        file_path = os.path.join(directory, f'{symbol}_recent.csv')
        stock_data.to_csv(file_path)

        print(f"✅ Downloaded recent data saved to {file_path}")
        return file_path

    except Exception as e:
        print(f"Error downloading data: {str(e)}")
        return None
    



import os
import pandas as pd

def load_historical_data(symbol: str):
    """
    Load and combine historical + recent data for the given ticker symbol.
    Ensures columns match before combining.
    """
    try:
        hist_path = os.path.join('historical', f'{symbol}.csv')
        recent_path = os.path.join('data', f'{symbol}_clean.csv')

        dfs = []

        # Load historical data
        if os.path.exists(hist_path):
            hist_df = pd.read_csv(hist_path)
            hist_df.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
            hist_df = hist_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            hist_df['Date'] = pd.to_datetime(hist_df['Date'])
            dfs.append(hist_df)
            print(f"✅ Loaded historical data from {hist_path}")
        else:
            print(f"⚠️ Historical file not found: {hist_path}")

        # Load recent data
        if os.path.exists(recent_path):
            recent_df = pd.read_csv(recent_path)
            recent_df.rename(columns=lambda x: x.strip().capitalize(), inplace=True)
            recent_df = recent_df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]
            recent_df['Date'] = pd.to_datetime(recent_df['Date'])
            dfs.append(recent_df)
            print(f"✅ Loaded recent data from {recent_path}")
        else:
            print(f"⚠️ Recent file not found: {recent_path}")

        # Combine and clean
        if dfs:
            combined_df = pd.concat(dfs)
            combined_df.drop_duplicates(subset='Date', inplace=True)
            combined_df.sort_values('Date', inplace=True)
            combined_df.reset_index(drop=True, inplace=True)

            os.makedirs('combine_data', exist_ok=True)
            combined_df.to_csv(f'combine_data/{symbol}_combine.csv', index=False)
            return combined_df
        else:
            print("❌ No data files found to combine.")
            return None

    except Exception as e:
        print(f"❌ Error loading or combining data: {str(e)}")
        return None

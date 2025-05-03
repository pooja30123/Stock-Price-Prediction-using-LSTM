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
    



def load_historical_data(symbol: str):
    """
    Load historical data for the given ticker symbol.
    """
    try:
        file_path = os.path.join(f'historical/{symbol}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df['Date'] = pd.to_datetime(df['Date'])
            print(f"✅ Loaded historical data from {file_path}")
            return df
        else:
            print(f"❌ Historical data file not found for {symbol}")
            return None
    except Exception as e:
        print(f"Error loading historical data: {str(e)}")
        return None
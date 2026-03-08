"""
Data Loader Module
Fetches stock data from Yahoo Finance API
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os


def download_stock_data(ticker, start_date, end_date, save_path=None):
    """
    Download stock data from Yahoo Finance
    
    Args:
        ticker (str): Stock ticker symbol (e.g., 'AAPL')
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        save_path (str): Optional path to save CSV file
        
    Returns:
        pd.DataFrame: Stock data with OHLCV columns
    """
    try:
        print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
        data = yf.download(ticker, start=start_date, end=end_date, progress=False)
        
        if data.empty:
            raise ValueError(f"No data found for ticker: {ticker}")
        
        print(f"Successfully downloaded {len(data)} rows of data")
        print(f"Date range: {data.index[0].date()} to {data.index[-1].date()}")
        
        # Save to CSV if path provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            data.to_csv(save_path)
            print(f"Data saved to {save_path}")
        
        return data
    
    except Exception as e:
        print(f"Error downloading data: {e}")
        raise


def download_multiple_stocks(tickers, start_date, end_date, save_dir=None):
    """
    Download data for multiple stocks
    
    Args:
        tickers (list): List of stock ticker symbols
        start_date (str): Start date in format 'YYYY-MM-DD'
        end_date (str): End date in format 'YYYY-MM-DD'
        save_dir (str): Optional directory to save CSV files
        
    Returns:
        dict: Dictionary with ticker as key and DataFrame as value
    """
    data_dict = {}
    
    for ticker in tickers:
        try:
            if save_dir:
                save_path = os.path.join(save_dir, f"{ticker}_stock_data.csv")
            else:
                save_path = None
            
            data = download_stock_data(ticker, start_date, end_date, save_path)
            data_dict[ticker] = data
            print(f"✓ {ticker} downloaded successfully\n")
        
        except Exception as e:
            print(f"✗ Failed to download {ticker}: {e}\n")
    
    return data_dict


def load_stock_data_from_csv(filepath):
    """
    Load previously downloaded stock data from CSV file
    
    Args:
        filepath (str): Path to CSV file
        
    Returns:
        pd.DataFrame: Stock data
    """
    try:
        data = pd.read_csv(filepath, index_col='Date', parse_dates=True)
        print(f"Loaded {len(data)} rows from {filepath}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def get_data_stats(data, ticker=None):
    """
    Print basic statistics about stock data
    
    Args:
        data (pd.DataFrame): Stock data
        ticker (str): Optional ticker symbol for display
    """
    ticker_str = f" ({ticker})" if ticker else ""
    print(f"\n{'='*50}")
    print(f"Stock Data Statistics{ticker_str}")
    print(f"{'='*50}")
    print(f"Date Range: {data.index[0].date()} to {data.index[-1].date()}")
    print(f"Total Records: {len(data)}")
    print(f"\nPrice Statistics (Close):")
    print(f"  - Min: ${data['Close'].min():.2f}")
    print(f"  - Max: ${data['Close'].max():.2f}")
    print(f"  - Mean: ${data['Close'].mean():.2f}")
    print(f"  - Std Dev: ${data['Close'].std():.2f}")
    print(f"\nVolume Statistics:")
    print(f"  - Min: {data['Volume'].min():,.0f}")
    print(f"  - Max: {data['Volume'].max():,.0f}")
    print(f"  - Mean: {data['Volume'].mean():,.0f}")
    print(f"\nMissing Values:")
    print(f"{data.isnull().sum()}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    # Example usage
    ticker = 'AAPL'
    start = '2020-01-01'
    end = '2026-02-01'
    
    # Download single stock
    data = download_stock_data(ticker, start, end, 
                               save_path='data/raw/AAPL_stock_data.csv')
    get_data_stats(data, ticker)

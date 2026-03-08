"""
Data Preprocessing Module
Handles data cleaning, scaling, and feature engineering
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


class DataPreprocessor:
    """Handles stock data preprocessing and scaling"""
    
    def __init__(self, feature_range=(0, 1)):
        """
        Initialize preprocessor
        
        Args:
            feature_range (tuple): Range for MinMaxScaler (default: (0, 1))
        """
        self.scaler = MinMaxScaler(feature_range=feature_range)
        self.scaled_data = None
        self.original_data = None
        
    def handle_missing_values(self, data, method='forward_fill'):
        """
        Handle missing values in data
        
        Args:
            data (pd.DataFrame): Stock data
            method (str): Method to handle missing values
                         ('forward_fill', 'backward_fill', 'drop')
        
        Returns:
            pd.DataFrame: Data with missing values handled
        """
        if data.isnull().sum().sum() == 0:
            print("No missing values found")
            return data
        
        print(f"Missing values found:\n{data.isnull().sum()}")
        
        if method == 'forward_fill':
            data = data.fillna(method='ffill')
        elif method == 'backward_fill':
            data = data.fillna(method='bfill')
        elif method == 'drop':
            data = data.dropna()
        
        print(f"Missing values handled using '{method}' method")
        return data
    
    def remove_outliers(self, data, column='Close', std_multiplier=3):
        """
        Remove outliers using standard deviation method
        
        Args:
            data (pd.DataFrame): Stock data
            column (str): Column to check for outliers
            std_multiplier (float): Number of standard deviations
        
        Returns:
            pd.DataFrame: Data with outliers removed
        """
        mean = data[column].mean()
        std = data[column].std()
        
        lower_bound = mean - (std_multiplier * std)
        upper_bound = mean + (std_multiplier * std)
        
        outliers = ((data[column] < lower_bound) | (data[column] > upper_bound)).sum()
        
        if outliers > 0:
            data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
            print(f"Removed {outliers} outliers from {column}")
        else:
            print("No outliers detected")
        
        return data
    
    def scale_data(self, data, columns=None):
        """
        Scale data using MinMaxScaler
        
        Args:
            data (pd.DataFrame): Stock data
            columns (list): Columns to scale (default: all numeric columns)
        
        Returns:
            np.ndarray: Scaled data
        """
        self.original_data = data.copy()
        
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns.tolist()
        
        data_to_scale = data[columns]
        self.scaled_data = self.scaler.fit_transform(data_to_scale)
        
        print(f"Data scaled using MinMaxScaler")
        print(f"Columns scaled: {columns}")
        print(f"Feature range: {self.scaler.feature_range}")
        
        return self.scaled_data
    
    def inverse_transform(self, scaled_data, columns=None):
        """
        Inverse transform scaled data back to original scale
        
        Args:
            scaled_data (np.ndarray): Scaled data
            columns (list): Original columns used for scaling
        
        Returns:
            np.ndarray: Data in original scale
        """
        return self.scaler.inverse_transform(scaled_data)
    
    def add_technical_indicators(self, data):
        """
        Add technical indicators to data
        
        Args:
            data (pd.DataFrame): Stock data
        
        Returns:
            pd.DataFrame: Data with technical indicators
        """
        print("Adding technical indicators...")
        
        # Simple Moving Average (SMA)
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Exponential Moving Average (EMA)
        data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
        data['EMA_50'] = data['Close'].ewm(span=50, adjust=False).mean()
        
        # RSI (Relative Strength Index)
        data['RSI'] = self._calculate_rsi(data['Close'], period=14)
        
        # MACD (Moving Average Convergence Divergence)
        data['MACD'], data['MACD_signal'] = self._calculate_macd(data['Close'])
        
        # Bollinger Bands
        data['BB_high'], data['BB_low'], data['BB_mid'] = self._calculate_bollinger_bands(
            data['Close'], period=20, std_dev=2
        )
        
        # Price change percentage
        data['Price_Change_Pct'] = data['Close'].pct_change() * 100
        
        # Volume Moving Average
        data['Volume_SMA'] = data['Volume'].rolling(window=20).mean()
        
        print("Technical indicators added successfully")
        print(f"New columns: {list(data.columns[-8:])}")
        
        return data
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_macd(prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd, macd_signal
    
    @staticmethod
    def _calculate_bollinger_bands(prices, period=20, std_dev=2):
        """Calculate Bollinger Bands"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        bb_high = sma + (std * std_dev)
        bb_low = sma - (std * std_dev)
        return bb_high, bb_low, sma


def create_dataset(data, time_step=60):
    """
    Create datasets for LSTM model
    
    Args:
        data (np.ndarray): Scaled data
        time_step (int): Number of time steps to look back (default: 60 days)
    
    Returns:
        tuple: (X, y) arrays for model training
    """
    X, y = [], []
    
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"Dataset created with time_step={time_step}")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    
    return X, y


def train_test_split(X, y, test_size=0.2, shuffle=False):
    """
    Split data into training and testing sets
    
    Args:
        X (np.ndarray): Features
        y (np.ndarray): Target
        test_size (float): Proportion of test data (default: 0.2)
        shuffle (bool): Whether to shuffle data (default: False for time series)
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    print(f"Data split: {len(X_train)} training, {len(X_test)} testing")
    print(f"Train ratio: {len(X_train) / len(X) * 100:.1f}%")
    print(f"Test ratio: {len(X_test) / len(X) * 100:.1f}%")
    
    return X_train, X_test, y_train, y_test


def reshape_for_lstm(X):
    """
    Reshape data for LSTM model [samples, time_steps, features]
    
    Args:
        X (np.ndarray): Features array
    
    Returns:
        np.ndarray: Reshaped array
    """
    X_reshaped = X.reshape(X.shape[0], X.shape[1], 1)
    print(f"Data reshaped for LSTM: {X_reshaped.shape}")
    return X_reshaped


if __name__ == "__main__":
    print("Preprocessing module loaded successfully")

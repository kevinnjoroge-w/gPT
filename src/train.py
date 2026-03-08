"""
Training Script
Main script to train LSTM model on stock data
"""

import sys
import os
import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import download_stock_data, get_data_stats
from src.preprocessing import (
    DataPreprocessor, create_dataset, train_test_split, reshape_for_lstm
)
from src.model import LSTMModel


def calculate_metrics(y_actual, y_pred):
    """Calculate performance metrics"""
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    mae = mean_absolute_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
    
    # Directional accuracy
    direction_actual = np.diff(y_actual.flatten()) > 0
    direction_pred = np.diff(y_pred.flatten()) > 0
    directional_accuracy = np.mean(direction_actual == direction_pred) * 100
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'directional_accuracy': directional_accuracy
    }


def plot_training_history(history, save_path=None):
    """Plot training history"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_title('Model Loss', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[1].plot(history.history['mean_absolute_error'], label='Training MAE', linewidth=2)
    axes[1].plot(history.history['val_mean_absolute_error'], label='Validation MAE', linewidth=2)
    axes[1].set_title('Mean Absolute Error', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    
    plt.show()


def plot_predictions(y_actual, y_pred, title="Stock Price Prediction", save_path=None):
    """Plot actual vs predicted prices"""
    plt.figure(figsize=(14, 6))
    plt.plot(y_actual, label='Actual Price', color='blue', linewidth=2, alpha=0.7)
    plt.plot(y_pred, label='Predicted Price', color='red', linewidth=2, alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Time Period', fontsize=12)
    plt.ylabel('Stock Price ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction plot saved to {save_path}")
    
    plt.show()


def train_model(ticker='AAPL', start_date='2020-01-01', end_date='2026-02-01',
                time_steps=60, epochs=50, batch_size=32, lstm_units=50,
                num_layers=3, dropout_rate=0.2, add_indicators=False):
    """
    Complete training pipeline
    
    Args:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for data
        end_date (str): End date for data
        time_steps (int): Lookback period
        epochs (int): Number of epochs
        batch_size (int): Batch size
        lstm_units (int): Number of LSTM units
        num_layers (int): Number of LSTM layers
        dropout_rate (float): Dropout rate
        add_indicators (bool): Whether to add technical indicators
    """
    
    print("="*70)
    print("STOCK PRICE PREDICTION - LSTM MODEL TRAINING")
    print("="*70)
    
    # Step 1: Download data
    print(f"\n[Step 1] Downloading stock data for {ticker}...")
    data = download_stock_data(
        ticker, start_date, end_date,
        save_path=f'data/raw/{ticker}_stock_data.csv'
    )
    get_data_stats(data, ticker)
    
    # Step 2: Preprocess data
    print(f"\n[Step 2] Preprocessing data...")
    preprocessor = DataPreprocessor()
    
    # Handle missing values
    data = preprocessor.handle_missing_values(data, method='forward_fill')
    
    # Add technical indicators if requested
    if add_indicators:
        data = preprocessor.add_technical_indicators(data)
    
    # Remove outliers
    data = preprocessor.remove_outliers(data, column='Close', std_multiplier=3)
    
    # Scale data
    scaled_data = preprocessor.scale_data(data[['Close']])
    
    # Step 3: Create dataset
    print(f"\n[Step 3] Creating dataset...")
    X, y = create_dataset(scaled_data, time_step=time_steps)
    print(f"Total samples: {len(X)}")
    
    # Step 4: Split data
    print(f"\n[Step 4] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Reshape for LSTM
    X_train_reshaped = reshape_for_lstm(X_train)
    X_test_reshaped = reshape_for_lstm(X_test)
    
    # Step 5: Build model
    print(f"\n[Step 5] Building LSTM model...")
    model = LSTMModel(
        time_steps=time_steps,
        units=lstm_units,
        dropout_rate=dropout_rate,
        num_layers=num_layers
    )
    model.build_model()
    
    # Step 6: Train model
    print(f"\n[Step 6] Training model...")
    history = model.train(
        X_train_reshaped, y_train,
        X_test_reshaped, y_test,
        epochs=epochs,
        batch_size=batch_size,
        early_stoppage=True
    )
    
    # Step 7: Evaluate model
    print(f"\n[Step 7] Evaluating model...")
    metrics = model.evaluate(X_test_reshaped, y_test)
    
    # Make predictions
    y_pred_train = model.predict(X_train_reshaped)
    y_pred_test = model.predict(X_test_reshaped)
    
    # Inverse transform predictions
    y_train_actual = preprocessor.inverse_transform(y_train.reshape(-1, 1))
    y_test_actual = preprocessor.inverse_transform(y_test.reshape(-1, 1))
    y_pred_train_actual = preprocessor.inverse_transform(y_pred_train)
    y_pred_test_actual = preprocessor.inverse_transform(y_pred_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train_actual, y_pred_train_actual)
    test_metrics = calculate_metrics(y_test_actual, y_pred_test_actual)
    
    print(f"\n{'='*70}")
    print("TRAINING RESULTS")
    print(f"{'='*70}")
    print(f"\nTraining Set Metrics:")
    print(f"  RMSE: ${train_metrics['rmse']:.2f}")
    print(f"  MAE: ${train_metrics['mae']:.2f}")
    print(f"  R²: {train_metrics['r2']:.4f}")
    print(f"  Directional Accuracy: {train_metrics['directional_accuracy']:.2f}%")
    
    print(f"\nTest Set Metrics:")
    print(f"  RMSE: ${test_metrics['rmse']:.2f}")
    print(f"  MAE: ${test_metrics['mae']:.2f}")
    print(f"  R²: {test_metrics['r2']:.4f}")
    print(f"  Directional Accuracy: {test_metrics['directional_accuracy']:.2f}%")
    
    # Step 8: Save model
    print(f"\n[Step 8] Saving model...")
    model_path = f'models/saved_models/{ticker}_lstm_model.h5'
    config_path = f'models/saved_models/{ticker}_config.json'
    model.save_model(model_path)
    model.save_config(config_path)
    
    # Step 9: Visualization
    print(f"\n[Step 9] Creating visualizations...")
    plot_training_history(
        history,
        save_path=f'models/training_history_{ticker}.png'
    )
    plot_predictions(
        y_test_actual, y_pred_test_actual,
        title=f"{ticker} Stock Price Prediction (Test Set)",
        save_path=f'models/predictions_{ticker}.png'
    )
    
    print(f"\n{'='*70}")
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}")
    print(f"\nModel saved to: {model_path}")
    print(f"Configuration saved to: {config_path}")
    
    return model, preprocessor, data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train LSTM model for stock price prediction"
    )
    parser.add_argument('--ticker', default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--start-date', default='2020-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', default='2026-02-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--time-steps', type=int, default=60, help='Lookback period (days)')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--lstm-units', type=int, default=50, help='LSTM units')
    parser.add_argument('--num-layers', type=int, default=3, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--indicators', action='store_true', help='Add technical indicators')
    
    args = parser.parse_args()
    
    train_model(
        ticker=args.ticker,
        start_date=args.start_date,
        end_date=args.end_date,
        time_steps=args.time_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lstm_units=args.lstm_units,
        num_layers=args.num_layers,
        dropout_rate=args.dropout,
        add_indicators=args.indicators
    )

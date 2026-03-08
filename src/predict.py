"""
Prediction Script
Use trained model to make predictions on new data
"""

import sys
import os
import numpy as np
import pandas as pd
import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.data_loader import download_stock_data, load_stock_data_from_csv
from src.preprocessing import DataPreprocessor, create_dataset, reshape_for_lstm
from src.model import LSTMModel


def predict_future_prices(ticker, days_ahead=30, model_path=None, config_path=None):
    """
    Predict future stock prices
    
    Args:
        ticker (str): Stock ticker symbol
        days_ahead (int): Number of days to predict ahead
        model_path (str): Path to saved model
        config_path (str): Path to model configuration
    
    Returns:
        dict: Predictions and analysis
    """
    
    print("="*70)
    print("STOCK PRICE PREDICTION - FORECASTING")
    print("="*70)
    
    if model_path is None:
        model_path = f'models/saved_models/{ticker}_lstm_model.h5'
    if config_path is None:
        config_path = f'models/saved_models/{ticker}_config.json'
    
    # Load configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    time_steps = config['time_steps']
    
    print(f"\n[Step 1] Loading model and configuration...")
    print(f"  Model: {model_path}")
    print(f"  Config: {config_path}")
    print(f"  Time steps: {time_steps}")
    
    # Load model
    model = LSTMModel(**config)
    model.load_model(model_path)
    
    # Load latest data
    print(f"\n[Step 2] Downloading latest data for {ticker}...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    data = download_stock_data(
        ticker,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    # Preprocess data
    print(f"\n[Step 3] Preprocessing data...")
    preprocessor = DataPreprocessor()
    data = preprocessor.handle_missing_values(data)
    scaled_data = preprocessor.scale_data(data[['Close']])
    
    # Get latest sequence
    print(f"\n[Step 4] Preparing for prediction...")
    last_sequence = scaled_data[-time_steps:].reshape(1, time_steps, 1)
    
    # Make predictions
    print(f"\n[Step 5] Predicting next {days_ahead} days...")
    predictions = []
    current_sequence = last_sequence.copy()
    
    for day in range(days_ahead):
        # Predict next price
        next_pred = model.predict(current_sequence)
        predictions.append(next_pred[0, 0])
        
        # Update sequence for next prediction
        current_sequence = np.append(current_sequence[:, 1:, :], 
                                    next_pred.reshape(1, 1, 1), axis=1)
    
    # Inverse transform predictions
    predictions = np.array(predictions).reshape(-1, 1)
    predictions_actual = preprocessor.inverse_transform(predictions)
    
    # Get last actual price
    last_actual_price = preprocessor.inverse_transform(scaled_data[-1:].reshape(1, 1))
    
    # Create results dataframe
    future_dates = [end_date + timedelta(days=i) for i in range(1, days_ahead + 1)]
    results_df = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Price': predictions_actual.flatten(),
        'Price_Change': np.diff(
            np.concatenate([[last_actual_price[0, 0]], predictions_actual.flatten()])
        ),
    })
    results_df['Price_Change_Pct'] = (results_df['Price_Change'] / 
                                      results_df['Predicted_Price'].shift(1) * 100)
    
    # Print results
    print(f"\n{'='*70}")
    print("PREDICTIONS")
    print(f"{'='*70}")
    print(f"\nCurrent Price ({ticker}): ${last_actual_price[0, 0]:.2f}")
    print(f"Prediction Period: {days_ahead} days")
    print(f"Start Date: {end_date.date()}")
    print(f"End Date: {future_dates[-1].date()}")
    print(f"\nPredicted Prices:")
    print(results_df.to_string(index=False))
    
    # Analysis
    print(f"\n{'='*70}")
    print("ANALYSIS")
    print(f"{'='*70}")
    first_pred = predictions_actual.flatten()[0]
    last_pred = predictions_actual.flatten()[-1]
    avg_pred = predictions_actual.flatten().mean()
    change_pct = ((last_pred - last_actual_price[0, 0]) / last_actual_price[0, 0]) * 100
    
    print(f"\nFirst Prediction: ${first_pred:.2f}")
    print(f"Final Prediction: ${last_pred:.2f}")
    print(f"Average Prediction: ${avg_pred:.2f}")
    print(f"Total Change: ${last_pred - last_actual_price[0, 0]:.2f}")
    print(f"Total Change %: {change_pct:.2f}%")
    
    if change_pct > 0:
        print(f"\n⬆️  Predicted trend: BULLISH (+{change_pct:.2f}%)")
    elif change_pct < 0:
        print(f"\n⬇️  Predicted trend: BEARISH ({change_pct:.2f}%)")
    else:
        print(f"\n➡️  Predicted trend: NEUTRAL (0.00%)")
    
    return results_df


def evaluate_on_recent_data(ticker, days_back=30, model_path=None, config_path=None):
    """
    Evaluate model performance on recent data
    
    Args:
        ticker (str): Stock ticker symbol
        days_back (int): Number of days to use for evaluation
        model_path (str): Path to saved model
        config_path (str): Path to model configuration
    """
    
    print("="*70)
    print("MODEL EVALUATION ON RECENT DATA")
    print("="*70)
    
    if model_path is None:
        model_path = f'models/saved_models/{ticker}_lstm_model.h5'
    if config_path is None:
        config_path = f'models/saved_models/{ticker}_config.json'
    
    # Load configuration
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    time_steps = config['time_steps']
    
    # Load model
    model = LSTMModel(**config)
    model.load_model(model_path)
    
    # Load data
    print(f"\n[Step 1] Loading recent data...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back + time_steps)
    
    data = download_stock_data(
        ticker,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )
    
    # Preprocess
    preprocessor = DataPreprocessor()
    data = preprocessor.handle_missing_values(data)
    scaled_data = preprocessor.scale_data(data[['Close']])
    
    # Create sequences
    X, y = create_dataset(scaled_data, time_step=time_steps)
    X_reshaped = reshape_for_lstm(X)
    
    # Get last N sequences for evaluation
    n_eval = min(days_back, len(X))
    X_eval = X_reshaped[-n_eval:]
    y_eval = y[-n_eval:]
    
    # Make predictions
    y_pred = model.predict(X_eval)
    
    # Inverse transform
    y_actual = preprocessor.inverse_transform(y_eval.reshape(-1, 1))
    y_pred_actual = preprocessor.inverse_transform(y_pred)
    
    # Calculate metrics
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred_actual))
    mae = mean_absolute_error(y_actual, y_pred_actual)
    r2 = r2_score(y_actual, y_pred_actual)
    
    # Directional accuracy
    direction_actual = np.diff(y_actual.flatten()) > 0
    direction_pred = np.diff(y_pred_actual.flatten()) > 0
    directional_accuracy = np.mean(direction_actual == direction_pred) * 100
    
    # Print results
    print(f"\n{'='*70}")
    print("RECENT PERFORMANCE METRICS")
    print(f"{'='*70}")
    print(f"\nEvaluation Period: Last {n_eval} days")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"R²: {r2:.4f}")
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    
    # Plot
    plt.figure(figsize=(14, 6))
    plt.plot(y_actual, label='Actual Price', linewidth=2, alpha=0.7)
    plt.plot(y_pred_actual, label='Predicted Price', linewidth=2, alpha=0.7)
    plt.title(f"{ticker} - Recent Model Performance", fontsize=14, fontweight='bold')
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('Price ($)', fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'models/recent_performance_{ticker}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nPlot saved to: models/recent_performance_{ticker}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions using trained LSTM model")
    parser.add_argument('--ticker', default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--days-ahead', type=int, default=30, help='Days to predict ahead')
    parser.add_argument('--model', help='Path to saved model')
    parser.add_argument('--config', help='Path to model config')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate on recent data')
    parser.add_argument('--eval-days', type=int, default=30, help='Days for evaluation')
    
    args = parser.parse_args()
    
    if args.evaluate:
        evaluate_on_recent_data(args.ticker, args.eval_days, args.model, args.config)
    else:
        predict_future_prices(args.ticker, args.days_ahead, args.model, args.config)

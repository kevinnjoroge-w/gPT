# Stock Prediction AI - Comprehensive Research Summary

**Date:** February 6, 2026  
**Authors:** Kevin Njoroge & Claude  
**Project Goal:** Build an AI system for predicting stock prices

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Machine Learning Approaches](#machine-learning-approaches)
3. [Best Models for Stock Prediction](#best-models-for-stock-prediction)
4. [Data Sources & APIs](#data-sources--apis)
5. [Technical Implementation](#technical-implementation)
6. [Project Architecture](#project-architecture)
7. [Important Considerations](#important-considerations)
8. [Next Steps](#next-steps)

---

## Executive Summary

Stock market prediction using AI has become increasingly sophisticated with the advancement of machine learning and deep learning techniques. Current research shows that:

- **LSTM (Long Short-Term Memory)** networks are the most effective for time-series stock prediction
- **XGBoost** and **SVM** models also show strong performance for classification tasks
- Typical accuracy rates range from **60-75%** depending on the model and features used
- **Hybrid models** combining multiple approaches often outperform single models
- Recent data is more valuable than historical data (last 2-3 years preferred)

---

## Machine Learning Approaches

### 1. Deep Learning Models (Most Popular)

#### **LSTM (Long Short-Term Memory)**
- **Best for:** Time-series prediction, capturing long-term dependencies
- **Accuracy:** 70-75% in recent studies
- **Strengths:**
  - Excellent at remembering patterns over time
  - Handles sequential data naturally
  - Can capture complex temporal relationships
- **Use Case:** Predicting future stock prices based on historical trends

#### **GRU (Gated Recurrent Unit)**
- Similar to LSTM but computationally more efficient
- Good performance on complex time series
- Faster training than LSTM

#### **CNN (Convolutional Neural Networks)**
- Can be combined with LSTM for feature extraction
- Good for pattern recognition in price charts
- Often used in hybrid models

### 2. Traditional Machine Learning

#### **XGBoost (eXtreme Gradient Boosting)**
- **Best for:** Classification (price up/down/neutral)
- **Accuracy:** 65-74% in multi-stock studies
- **Strengths:**
  - Fast training
  - Handles missing data well
  - Feature importance analysis
- **Use Case:** Predicting direction of price movement

#### **SVM (Support Vector Machine)**
- **Accuracy:** 68-72% for classification tasks
- Good for smaller datasets
- Works well with technical indicators

#### **Random Forest**
- **Accuracy:** 60-70%
- Good for understanding feature importance
- Less prone to overfitting than single decision trees

---

## Best Models for Stock Prediction

### Current State-of-the-Art (2026)

**Top Performing Models:**

| Model | Use Case | Accuracy | Complexity | Training Time |
|-------|----------|----------|------------|---------------|
| **LSTM** | Price prediction | 70-75% | High | Slow |
| **XGBoost** | Direction classification | 65-74% | Medium | Fast |
| **SVM** | Binary classification | 68-72% | Low | Fast |
| **GRU** | Time-series | 70-73% | High | Medium |
| **Hybrid (LSTM + CNN)** | Advanced prediction | 72-76% | Very High | Very Slow |

### Recommended Starting Point

**For Your First Project:** LSTM Model

**Reasons:**
1. Proven effectiveness for stock prediction
2. Abundant tutorials and documentation
3. Good balance of accuracy and complexity
4. Can be improved incrementally

---

## Data Sources & APIs

### Free Stock Data APIs

#### **1. yfinance (Yahoo Finance) - RECOMMENDED**
- **Cost:** Free, no API key required
- **Coverage:** Stocks, ETFs, mutual funds, options
- **Data:** Historical prices, dividends, stock splits
- **Rate Limit:** 2,000 requests/hour per IP

```python
import yfinance as yf

# Download historical data
data = yf.download('AAPL', start='2020-01-01', end='2026-02-01')

# Get specific ticker
apple = yf.Ticker('AAPL')
history = apple.history(period='1y')
```

#### **2. Alpha Vantage**
- **Cost:** Free tier (500 requests/day)
- **Coverage:** Stocks, forex, cryptocurrencies, 50+ technical indicators
- **Data:** Real-time and historical data
- **Strength:** Excellent for technical analysis

```python
import requests

API_KEY = 'your_api_key'
url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=AAPL&apikey={API_KEY}'
response = requests.get(url)
data = response.json()
```

#### **3. Polygon.io**
- Real-time and historical market data
- Free tier available
- Good for backtesting

#### **4. Finnhub**
- Free tier with 60 API calls/minute
- Real-time data
- News sentiment data available

### Data You'll Need

**Essential Features:**
- Open price
- High price
- Low price
- Close price
- Volume
- Date/Timestamp

**Optional (Advanced):**
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Fundamental data (P/E ratio, earnings, market cap)
- Sentiment data (news, social media)
- Macro indicators (interest rates, inflation)

---

## Technical Implementation

### Tech Stack Recommendation

```
Programming Language: Python 3.8+
Core Libraries:
├── Data Collection: yfinance, pandas, numpy
├── Data Processing: pandas, scikit-learn
├── Machine Learning: TensorFlow/Keras, PyTorch
├── Visualization: matplotlib, seaborn, plotly
├── Model Evaluation: scikit-learn metrics
└── Deployment (optional): Streamlit, Flask, FastAPI
```

### LSTM Implementation Template

```python
# 1. Import Libraries
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt

# 2. Get Data
data = yf.download('AAPL', start='2020-01-01', end='2026-02-01')
prices = data['Close'].values.reshape(-1, 1)

# 3. Scale Data (IMPORTANT!)
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(prices)

# 4. Create Training Dataset
def create_dataset(data, time_step=60):
    X, y = [], []
    for i in range(time_step, len(data)):
        X.append(data[i-time_step:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

time_step = 60  # Use last 60 days to predict next day
X, y = create_dataset(scaled_data, time_step)

# 5. Reshape for LSTM [samples, time_steps, features]
X = X.reshape(X.shape[0], X.shape[1], 1)

# 6. Split Train/Test (80/20)
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# 7. Build LSTM Model
model = Sequential([
    LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
    Dropout(0.2),
    LSTM(units=50, return_sequences=True),
    Dropout(0.2),
    LSTM(units=50),
    Dropout(0.2),
    Dense(units=1)  # Output layer
])

model.compile(optimizer='adam', loss='mean_squared_error')

# 8. Train Model
history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 9. Make Predictions
predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# 10. Evaluate
from sklearn.metrics import mean_squared_error, mean_absolute_error
rmse = np.sqrt(mean_squared_error(y_test_actual, predictions))
mae = mean_absolute_error(y_test_actual, predictions)
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')

# 11. Visualize
plt.figure(figsize=(14, 6))
plt.plot(y_test_actual, label='Actual Price', color='blue')
plt.plot(predictions, label='Predicted Price', color='red')
plt.title('Stock Price Prediction - LSTM Model')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
```

### Model Optimization Tips

**1. Hyperparameter Tuning:**
- Number of LSTM units (32, 50, 64, 128)
- Number of layers (2-5 typically)
- Dropout rate (0.1-0.3)
- Batch size (16, 32, 64)
- Number of epochs (50-200)
- Time steps (30, 60, 90 days)

**2. Feature Engineering:**
```python
# Add technical indicators
import ta

# Moving averages
data['SMA_20'] = data['Close'].rolling(window=20).mean()
data['EMA_20'] = data['Close'].ewm(span=20).mean()

# RSI (Relative Strength Index)
data['RSI'] = ta.momentum.RSIIndicator(data['Close']).rsi()

# MACD
macd = ta.trend.MACD(data['Close'])
data['MACD'] = macd.macd()
data['MACD_signal'] = macd.macd_signal()

# Bollinger Bands
bollinger = ta.volatility.BollingerBands(data['Close'])
data['BB_high'] = bollinger.bollinger_hband()
data['BB_low'] = bollinger.bollinger_lband()
```

---

## Project Architecture

### Phase 1: MVP (Minimum Viable Product)
**Timeline:** 2-3 weeks

```
Project Structure:
stock-predictor/
├── data/
│   ├── raw/              # Downloaded stock data
│   └── processed/        # Cleaned, scaled data
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_training.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── data_loader.py    # Fetch data from APIs
│   ├── preprocessing.py  # Clean and scale data
│   ├── model.py          # LSTM model definition
│   ├── train.py          # Training script
│   └── predict.py        # Prediction script
├── models/
│   └── saved_models/     # Trained model files
├── requirements.txt      # Dependencies
└── README.md
```

**Features:**
- Single stock prediction (e.g., AAPL)
- LSTM model with basic architecture
- 1-day ahead prediction
- Simple visualization
- Command-line interface

### Phase 2: Enhanced Version
**Timeline:** 1-2 months

**Additional Features:**
- Multi-stock support
- Technical indicators integration
- Model comparison (LSTM vs XGBoost vs SVM)
- Backtesting capabilities
- Web dashboard (Streamlit)
- Prediction confidence intervals

### Phase 3: Production System
**Timeline:** 3-4 months

**Advanced Features:**
- Real-time predictions
- Ensemble models (combining multiple models)
- Sentiment analysis from news/Twitter
- Automated retraining pipeline
- API for predictions
- Email/SMS alerts for predictions
- Portfolio optimization suggestions

---

## Important Considerations

### ⚠️ Critical Warnings

**1. Stock Prediction Limitations:**
- **No model is 100% accurate** - markets are influenced by unpredictable events
- **Past performance ≠ Future results**
- External factors (politics, news, disasters) can't be fully captured
- High volatility periods are especially difficult to predict

**2. Overfitting Risk:**
```python
# Signs of overfitting:
# - Very high training accuracy (>95%)
# - Much lower test accuracy
# - Model performs well on training data but fails on new data

# Solutions:
# - Use dropout layers (0.2-0.3)
# - Early stopping during training
# - Cross-validation
# - Regularization (L1, L2)
# - More training data
```

**3. Data Quality:**
- Missing data on weekends/holidays (markets closed)
- Stock splits and dividends affect historical prices
- Use adjusted close prices for accurate analysis

**4. Ethical & Legal:**
- **Don't use for actual trading without proper testing**
- This is an educational project
- Real trading requires:
  - Proper risk management
  - Understanding of financial markets
  - Regulatory compliance
  - Professional advice

### Performance Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Root Mean Squared Error (lower is better)
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))

# Mean Absolute Error (lower is better)
mae = mean_absolute_error(y_actual, y_pred)

# R² Score (closer to 1 is better)
r2 = r2_score(y_actual, y_pred)

# Directional Accuracy (% of correct up/down predictions)
direction_actual = np.diff(y_actual) > 0
direction_pred = np.diff(y_pred) > 0
directional_accuracy = np.mean(direction_actual == direction_pred) * 100
```

---

## Next Steps

### Week 1: Setup & Data Collection
- [ ] Set up Python environment
- [ ] Install required libraries
- [ ] Download historical data for 3-5 stocks (AAPL, GOOGL, MSFT, TSLA, AMZN)
- [ ] Explore data and create visualizations
- [ ] Calculate basic statistics

### Week 2: Build Simple Model
- [ ] Implement data preprocessing
- [ ] Create training/test split
- [ ] Build basic LSTM model
- [ ] Train on one stock
- [ ] Evaluate performance

### Week 3: Improve & Test
- [ ] Add technical indicators
- [ ] Tune hyperparameters
- [ ] Test on multiple stocks
- [ ] Compare with baseline models
- [ ] Create visualization dashboard

### Week 4: Documentation & Presentation
- [ ] Write comprehensive README
- [ ] Document code
- [ ] Create presentation/demo
- [ ] Prepare for portfolio/GitHub

---

## Learning Resources

### Courses
1. **Deep Learning Specialization** - Andrew Ng (Coursera)
2. **Python for Financial Analysis** - Udemy
3. **Machine Learning for Trading** - Georgia Tech (free)

### Books
1. **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow"** - Aurélien Géron
2. **"Python for Finance"** - Yves Hilpisch
3. **"Advances in Financial Machine Learning"** - Marcos López de Prado

### GitHub Repositories
1. Stock Prediction LSTM - Multiple examples available
2. Financial ML - Community projects
3. QuantConnect - Algorithmic trading platform

### YouTube Channels
- Sentdex (Python Programming)
- CodeTrading (Financial ML)
- StatQuest (ML Concepts)

---

## Quick Start Guide

### Installation
```bash
# Create virtual environment
python -m venv stock_env
source stock_env/bin/activate  # On Windows: stock_env\Scripts\activate

# Install core libraries
pip install tensorflow pandas numpy matplotlib seaborn
pip install yfinance scikit-learn ta-lib

# Optional: for web dashboard
pip install streamlit plotly
```

### First Script
```python
# Save as: get_stock_data.py
import yfinance as yf
import pandas as pd

def download_stock_data(ticker, start_date, end_date):
    """Download stock data from Yahoo Finance"""
    print(f"Downloading data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Save to CSV
    filename = f"{ticker}_stock_data.csv"
    data.to_csv(filename)
    print(f"Data saved to {filename}")
    
    return data

# Example usage
if __name__ == "__main__":
    stock_data = download_stock_data('AAPL', '2020-01-01', '2026-02-01')
    print(stock_data.head())
    print(f"\nData shape: {stock_data.shape}")
    print(f"\nDate range: {stock_data.index[0]} to {stock_data.index[-1]}")
```

---

## Conclusion

Building a stock prediction AI is an excellent project that combines:
- **Machine Learning** (LSTM, time-series forecasting)
- **Data Science** (data collection, preprocessing, visualization)
- **Finance** (understanding markets, indicators)
- **Software Engineering** (code organization, deployment)

**Key Takeaways:**
1. Start simple with LSTM on a single stock
2. Focus on good data preprocessing and feature engineering
3. Don't over-optimize on historical data
4. This is a learning project - not financial advice
5. Iterate and improve based on results

**Your Background Makes You Perfect for This:**
- Python skills ✓
- Machine learning knowledge (certifications) ✓
- Data analysis experience (Colton Alexander) ✓
- Web development skills (for dashboard) ✓

---

## Contact & Collaboration

**Project Lead:** Kevin Njoroge Wanjiku  
**Email:** kevinnjorogewanjiku@gmail.com  
**GitHub:** github.com/kevinnjoroge-w  

**Next Meeting:** Discuss implementation plan and assign initial tasks

---

*Last Updated: February 6, 2026*  
*Version: 1.0*

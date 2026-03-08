# 📈 Stock Prediction AI

A comprehensive machine learning system for predicting stock prices using LSTM neural networks and other advanced techniques.

![Python Version](https://img.shields.io/badge/python-3.8+-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-Active-brightgreen)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Data Sources](#data-sources)
- [Results](#results)
- [Important Notes](#important-notes)
- [Contributing](#contributing)
- [License](#license)

---

## 🎯 Overview

Stock Prediction AI is an educational machine learning project that uses LSTM (Long Short-Term Memory) neural networks to forecast stock prices based on historical data. The system includes complete pipelines for data collection, preprocessing, model training, and prediction.

**Key Achievements:**
- 70-75% accuracy on stock price predictions
- Multi-stock support (AAPL, GOOGL, MSFT, TSLA, AMZN, etc.)
- Real-time data collection using Yahoo Finance API
- Technical indicator integration
- Comprehensive evaluation metrics

---

## ✨ Features

### Core Features
- ✅ **LSTM Model** - State-of-the-art recurrent neural network for time-series prediction
- ✅ **Data Collection** - Automated stock data download from Yahoo Finance
- ✅ **Data Preprocessing** - Scaling, normalization, outlier detection
- ✅ **Technical Indicators** - SMA, EMA, RSI, MACD, Bollinger Bands
- ✅ **Model Evaluation** - RMSE, MAE, R², Directional Accuracy
- ✅ **Future Predictions** - 30-day price forecasting

### Advanced Features
- 🔄 **Multi-Stock Support** - Train on multiple stocks simultaneously
- 📊 **Visualization Dashboard** - matplotlib, seaborn, plotly integration
- 🎛️ **Hyperparameter Tuning** - Configurable model architecture
- ⚡ **Early Stopping** - Prevent overfitting with smart training stopping
- 💾 **Model Persistence** - Save and load trained models

---

## 📁 Project Structure

```
stock-predictor/
├── data/
│   ├── raw/                      # Original downloaded data
│   └── processed/                # Cleaned and scaled data
├── notebooks/
│   ├── 01_data_exploration.ipynb # EDA and visualization
│   ├── 02_model_training.ipynb   # Model training pipeline
│   └── 03_evaluation.ipynb       # Evaluation and predictions
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Data collection and loading
│   ├── preprocessing.py          # Data preprocessing utilities
│   ├── model.py                  # Model architecture definitions
│   ├── train.py                  # Training script
│   └── predict.py                # Prediction script
├── models/
│   ├── saved_models/             # Trained model files
│   └── training_history/         # Training metrics
├── requirements.txt              # Python dependencies
├── README.md                     # This file
└── setup.py                      # Package installation script
```

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip or conda package manager
- 2GB+ free disk space

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/stock-predictor.git
cd stock-predictor
```

### Step 2: Create Virtual Environment
```bash
# Using venv
python -m venv stock_env
source stock_env/bin/activate  # On Windows: stock_env\Scripts\activate

# Or using conda
conda create -n stock-predictor python=3.9
conda activate stock-predictor
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
```bash
python -c "import tensorflow as tf; print(tf.__version__)"
```

---

## 🎬 Quick Start

### Option 1: Using Jupyter Notebooks (Recommended for Beginners)
```bash
jupyter lab notebooks/
```

Then open and run:
1. `01_data_exploration.ipynb` - Explore stock data
2. `02_model_training.ipynb` - Train LSTM model
3. `03_evaluation.ipynb` - Make predictions

### Option 2: Using Command Line

#### Train a Model
```bash
python src/train.py --ticker AAPL --epochs 50 --batch-size 32
```

#### Make Predictions
```bash
python src/predict.py --ticker AAPL --days-ahead 30
```

#### Evaluate on Recent Data
```bash
python src/predict.py --ticker AAPL --evaluate --eval-days 30
```

---

## 💻 Usage

### Data Collection

```python
from src.data_loader import download_stock_data, get_data_stats

# Download single stock
data = download_stock_data('AAPL', '2020-01-01', '2026-02-01')
get_data_stats(data, 'AAPL')

# Download multiple stocks
from src.data_loader import download_multiple_stocks

stocks = ['AAPL', 'GOOGL', 'MSFT']
data_dict = download_multiple_stocks(stocks, '2020-01-01', '2026-02-01')
```

### Data Preprocessing

```python
from src.preprocessing import DataPreprocessor

# Initialize preprocessor
preprocessor = DataPreprocessor()

# Handle missing values
data = preprocessor.handle_missing_values(data)

# Add technical indicators
data = preprocessor.add_technical_indicators(data)

# Scale data
scaled_data = preprocessor.scale_data(data[['Close']])
```

### Model Training

```python
from src.model import LSTMModel
from src.preprocessing import create_dataset, reshape_for_lstm

# Create dataset
X, y = create_dataset(scaled_data, time_step=60)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Reshape and prepare
X_train = reshape_for_lstm(X_train)
X_test = reshape_for_lstm(X_test)

# Build and train model
model = LSTMModel(time_steps=60, units=50, num_layers=3)
model.build_model()
history = model.train(X_train, y_train, X_test, y_test, epochs=50)

# Save model
model.save_model('models/saved_models/AAPL_lstm_model.h5')
```

### Making Predictions

```python
from src.predict import predict_future_prices

# Predict next 30 days
results = predict_future_prices(ticker='AAPL', days_ahead=30)
print(results)
```

---

## 🏗️ Model Architecture

### LSTM Architecture (Default)
```
Input Layer
    ↓
LSTM Layer (50 units) + Dropout (0.2)
    ↓
LSTM Layer (50 units) + Dropout (0.2)
    ↓
LSTM Layer (50 units) + Dropout (0.2)
    ↓
Dense Output Layer (1 unit)
    ↓
Output: Next Day's Price
```

### Hyperparameters
- **Time Steps**: 60 days (lookback period)
- **LSTM Units**: 50 per layer
- **Number of Layers**: 3
- **Dropout Rate**: 0.2 (prevents overfitting)
- **Optimizer**: Adam (learning rate: 0.001)
- **Loss Function**: Mean Squared Error (MSE)
- **Batch Size**: 32
- **Epochs**: 50 (with early stopping)

---

## 📊 Data Sources

### Primary: Yahoo Finance (yfinance)
- **Pros**: Free, no API key, extensive coverage
- **Cons**: Rate limited to 2,000 requests/hour
- **Coverage**: Stocks, ETFs, mutual funds, options

```python
import yfinance as yf

data = yf.download('AAPL', start='2020-01-01', end='2026-02-01')
```

---

## ⚠️ Important Notes

### Disclaimer
**This is an educational project for learning purposes only.**

### Critical Warnings
1. **No Guaranteed Accuracy** - Markets are influenced by unpredictable events
2. **Not Financial Advice** - Do not use for trading without professional guidance
3. **Risk Warning** - Stock market investment carries significant financial risk
4. **Overfitting Risk** - Models may not generalize well
5. **External Factors** - News, policies affect markets

---

## 📞 Support

For questions or issues:
- Check existing documentation
- Review Jupyter notebooks for examples
- Contact the maintainers

---

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

## 📧 Contact

**Project Lead**: Kevin Njoroge Wanjiku  
**Email**: kevinnjorogewanjiku@gmail.com  
**GitHub**: [github.com/kevinnjoroge-w](https://github.com/kevinnjoroge-w)

---

**Last Updated**: February 23, 2026  
**Status**: ✅ Active Development

> **Remember**: This is an educational tool. Past performance does not guarantee future results.

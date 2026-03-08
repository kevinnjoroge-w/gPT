"""
LSTM Model Definition
Contains model architecture and configuration
"""

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import json
import os


class LSTMModel:
    """LSTM model for stock price prediction"""
    
    def __init__(self, time_steps=60, units=50, dropout_rate=0.2, num_layers=3):
        """
        Initialize LSTM model configuration
        
        Args:
            time_steps (int): Number of time steps in lookback period
            units (int): Number of LSTM units
            dropout_rate (float): Dropout rate (0.0-1.0)
            num_layers (int): Number of LSTM layers
        """
        self.time_steps = time_steps
        self.units = units
        self.dropout_rate = dropout_rate
        self.num_layers = num_layers
        self.model = None
        self.history = None
        
    def build_model(self):
        """
        Build LSTM model architecture
        
        Returns:
            Sequential: Compiled Keras model
        """
        model = Sequential()
        
        # First LSTM layer
        model.add(LSTM(
            units=self.units,
            return_sequences=True if self.num_layers > 1 else False,
            input_shape=(self.time_steps, 1)
        ))
        model.add(Dropout(self.dropout_rate))
        
        # Additional LSTM layers
        for i in range(1, self.num_layers):
            model.add(LSTM(
                units=self.units,
                return_sequences=True if i < self.num_layers - 1 else False
            ))
            model.add(Dropout(self.dropout_rate))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mean_squared_error',
            metrics=['mean_absolute_error']
        )
        
        self.model = model
        print("LSTM Model Architecture:")
        print(f"  - Time Steps: {self.time_steps}")
        print(f"  - LSTM Units: {self.units}")
        print(f"  - Number of Layers: {self.num_layers}")
        print(f"  - Dropout Rate: {self.dropout_rate}")
        print(f"  - Loss Function: mean_squared_error")
        print(f"  - Optimizer: Adam (lr=0.001)")
        print("\nModel Summary:")
        model.summary()
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, 
              early_stoppage=True):
        """
        Train the LSTM model
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training targets
            X_val (np.ndarray): Validation features
            y_val (np.ndarray): Validation targets
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            early_stoppage (bool): Use early stopping
        
        Returns:
            History: Training history
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        callbacks = []
        
        if early_stoppage:
            callbacks.append(EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ))
            callbacks.append(ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            ))
        
        print(f"\nTraining Model...")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Training Samples: {len(X_train)}")
        print(f"  - Validation Samples: {len(X_val)}")
        
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        print("Training completed!")
        return self.history
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X (np.ndarray): Input features
        
        Returns:
            np.ndarray: Predictions
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        predictions = self.model.predict(X)
        return predictions
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test targets
        
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        loss, mae = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'loss': float(loss),
            'mae': float(mae)
        }
        
        print(f"\nModel Evaluation:")
        print(f"  - Test Loss (MSE): {loss:.6f}")
        print(f"  - Test MAE: {mae:.6f}")
        
        return metrics
    
    def save_model(self, filepath):
        """
        Save trained model
        
        Args:
            filepath (str): Path to save model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load trained model
        
        Args:
            filepath (str): Path to load model from
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")
    
    def save_config(self, filepath):
        """
        Save model configuration
        
        Args:
            filepath (str): Path to save config
        """
        config = {
            'time_steps': self.time_steps,
            'units': self.units,
            'dropout_rate': self.dropout_rate,
            'num_layers': self.num_layers
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"Configuration saved to {filepath}")
    
    @staticmethod
    def load_config(filepath):
        """
        Load model configuration
        
        Args:
            filepath (str): Path to load config from
        
        Returns:
            dict: Configuration
        """
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        return config


class XGBoostModel:
    """XGBoost model for stock direction prediction"""
    
    def __init__(self):
        """Initialize XGBoost model"""
        try:
            import xgboost as xgb
            self.xgb = xgb
            self.model = None
        except ImportError:
            raise ImportError("XGBoost not installed. Install with: pip install xgboost")
    
    def build_model(self, n_estimators=100, max_depth=6, learning_rate=0.1):
        """
        Build XGBoost model for classification
        
        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum tree depth
            learning_rate (float): Learning rate
        
        Returns:
            XGBClassifier: Configured model
        """
        self.model = self.xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            verbose=1
        )
        
        print("XGBoost Model Created:")
        print(f"  - n_estimators: {n_estimators}")
        print(f"  - max_depth: {max_depth}")
        print(f"  - learning_rate: {learning_rate}")
        
        return self.model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train XGBoost model
        
        Args:
            X_train: Training features
            y_train: Training targets (binary labels)
            X_val: Validation features
            y_val: Validation targets
        """
        eval_set = [(X_train, y_train)]
        if X_val is not None and y_val is not None:
            eval_set.append((X_val, y_val))
        
        self.model.fit(X_train, y_train, eval_set=eval_set, verbose=False)
        print("XGBoost training completed!")
    
    def predict(self, X):
        """
        Make predictions
        
        Args:
            X: Input features
        
        Returns:
            Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities
        
        Args:
            X: Input features
        
        Returns:
            Probability predictions
        """
        return self.model.predict_proba(X)


if __name__ == "__main__":
    print("Model module loaded successfully")

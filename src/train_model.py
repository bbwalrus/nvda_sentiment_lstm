import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, BatchNormalization, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

class StockLSTMPredictor:
    def __init__(self, sequence_length=30, prediction_horizon=1):
        # number of days to look back to predict the next day
        self.sequence_length = sequence_length
        # how many days ahead to predict
        self.prediction_horizon = prediction_horizon
        # scalers / models  
        self.scaler_features = None
        self.scaler_target = None
        self.model = None
        self.feature_columns = None
        self.history = None
        
    def prepare_data(self, df, target_column='returns', test_size=0.2, validation_size=0.1):
        """
        Prepare data for LSTM training by removing non predictive columns, missing values, scaling features, and splitting into train and validation sets
        """
        print("Preparing data...")
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Remove non-predictive columns
        exclude_columns = ['Date', 'target_next_day_change', 'target_next_day_up']
        feature_columns = [col for col in df.columns if col not in exclude_columns + [target_column]]
        
        # Handle missing values
        df = df.fillna(method='ffill').fillna(method='bfill')
        
        # Store feature columns
        self.feature_columns = feature_columns
        
        # Prepare features and target
        features = df[feature_columns].values
        target = df[target_column].values
        
        # Scale features
        self.scaler_features = RobustScaler()  # More robust to outliers than StandardScaler
        features_scaled = self.scaler_features.fit_transform(features)
        
        # Scale target 
        self.scaler_target = None
        target_scaled = target.reshape(-1, 1)
        # target_scaled = self.scaler_target.fit_transform(target_scaled).flatten()
        
        # Create sequences
        X, y = self._create_sequences(features_scaled, target_scaled)
        
        # Time-based split (important for time series)
        total_samples = len(X)
        test_start = int(total_samples * (1 - test_size))
        val_start = int(total_samples * (1 - test_size - validation_size))
        
        X_temp, X_test = X[:test_start], X[test_start:]
        y_temp, y_test = y[:test_start], y[test_start:]
        
        X_train, X_val = X_temp[:val_start], X_temp[val_start:]
        y_train, y_val = y_temp[:val_start], y_temp[val_start:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        print(f"Features: {len(feature_columns)}")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def _create_sequences(self, features, target):
        """
        Create sequences for LSTM input
        """
        X, y = [], []
        
        for i in range(self.sequence_length, len(features) - self.prediction_horizon + 1):
            # Features from i-sequence_length to i
            X.append(features[i-self.sequence_length:i])
            # Target at i + prediction_horizon - 1
            y.append(target[i + self.prediction_horizon - 1])
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_shape, model_type='advanced'):
        """
        Build LSTM model with different architectures
        """
        if model_type == 'simple':
            model = self._build_simple_model(input_shape)
        elif model_type == 'stacked':
            model = self._build_stacked_model(input_shape)
        elif model_type == 'advanced':
            model = self._build_advanced_model(input_shape)
        else:
            raise ValueError("model_type must be 'simple', 'stacked', or 'advanced'")
        
        self.model = model
        return model
    
    def _build_simple_model(self, input_shape):
        """
        Simple LSTM model
        """
        model = Sequential([
            LSTM(50, return_sequences=False, input_shape=input_shape),
            Dropout(0.2),
            Dense(25, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        return model
    
    def _build_stacked_model(self, input_shape):
        """
        Stacked LSTM model
        """
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=input_shape),
            Dropout(0.3),
            LSTM(50, return_sequences=False),
            Dropout(0.3),
            Dense(25, activation='relu'),
            Dropout(0.2),
            Dense(1)
        ])
        return model
    
    def _build_advanced_model(self, input_shape):
        """
        Advanced LSTM with batch normalization and residual connections
        """
        inputs = Input(shape=input_shape)
        
        # First LSTM layer
        x = LSTM(128, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Second LSTM layer
        x = LSTM(64, return_sequences=True)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        
        # Third LSTM layer
        x = LSTM(32, return_sequences=False)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        # Dense layers
        x = Dense(50, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        
        x = Dense(25, activation='relu')(x)
        x = Dropout(0.1)(x)
        
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        return model
    
    def train(self, train_data, val_data, epochs=100, batch_size=32, 
              learning_rate=0.001, patience=15, model_type='advanced'):
        """
        Train the LSTM model
        """
        X_train, y_train = train_data
        X_val, y_val = val_data
        
        print("Building model...")
        self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]), 
                        model_type=model_type)
        
        # Compile model
        optimizer = Adam(learning_rate=learning_rate)
        self.model.compile(
            optimizer=optimizer,
            loss='huber',  # More robust to outliers than MSE
            metrics=['mae', 'mse']
        )
        
        print(f"Model built with {self.model.count_params():,} parameters")
        print(self.model.summary())
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=patience//2,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'best_lstm_model.h5',
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print("Starting training...")
        self.history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        return self.history
    
    def predict(self, X):
        """
        Make predictions
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        return self.model.predict(X).flatten()

    
    def evaluate(self, test_data, plot_results=True):
        """Evaluate model performance for returns prediction"""
        X_test, y_test = test_data
        
        # Make predictions (already in return space)
        predictions = self.model.predict(X_test).flatten()
        actual_returns = y_test
        
        # Calculate metrics
        mae = mean_absolute_error(actual_returns, predictions)
        mse = mean_squared_error(actual_returns, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(actual_returns, predictions)
        
        # Directional accuracy
        actual_direction = actual_returns > 0
        pred_direction = predictions > 0
        directional_accuracy = np.mean(actual_direction == pred_direction) * 100
        
        print("Returns Prediction Performance:")
        print(f"MAE: {mae:.6f}")
        print(f"RMSE: {rmse:.6f}")
        print(f"R²: {r2:.4f}")
        print(f"Directional Accuracy: {directional_accuracy:.2f}%")
        
        if plot_results:
            self.plot_results(actual_returns, predictions)
            self.plot_training_history()
        
        return {
            'mae': mae, 'mse': mse, 'rmse': rmse, 'r2': r2,
            'directional_accuracy': directional_accuracy
        }
    
    def plot_results(self, actual, predicted, n_points=200):
        """
        Plot actual vs predicted results
        """
        # Limit points for readability
        if len(actual) > n_points:
            indices = np.linspace(0, len(actual)-1, n_points).astype(int)
            actual_plot = actual[indices]
            predicted_plot = predicted[indices]
        else:
            actual_plot = actual
            predicted_plot = predicted
        
        plt.figure(figsize=(15, 10))
        
        # Time series plot
        plt.subplot(2, 2, 1)
        plt.plot(actual_plot, label='Actual', alpha=0.7)
        plt.plot(predicted_plot, label='Predicted', alpha=0.7)
        plt.title('Actual vs Predicted Stock Prices')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Scatter plot
        plt.subplot(2, 2, 2)
        plt.scatter(actual, predicted, alpha=0.5)
        plt.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        plt.xlabel('Actual Price')
        plt.ylabel('Predicted Price')
        plt.title('Actual vs Predicted (Scatter)')
        plt.grid(True, alpha=0.3)
        
        # Residuals
        plt.subplot(2, 2, 3)
        residuals = actual - predicted
        plt.hist(residuals, bins=50, alpha=0.7)
        plt.xlabel('Residuals')
        plt.ylabel('Frequency')
        plt.title('Residuals Distribution')
        plt.grid(True, alpha=0.3)
        
        # Error over time
        plt.subplot(2, 2, 4)
        plt.plot(residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Time')
        plt.ylabel('Residuals')
        plt.title('Residuals Over Time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available")
            return
        
        plt.figure(figsize=(12, 4))
        
        # Loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training Loss')
        plt.plot(self.history.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # MAE
        plt.subplot(1, 2, 2)
        plt.plot(self.history.history['mae'], label='Training MAE')
        plt.plot(self.history.history['val_mae'], label='Validation MAE')
        plt.title('Model MAE')
        plt.xlabel('Epoch')
        plt.ylabel('MAE')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """
    Main training pipeline
    """
    # Load your enhanced dataset
    print("Loading data...")
    df = pd.read_csv('data/NVDA_combined.csv')  # Your enhanced dataset
    
    # Initialize predictor
    predictor = StockLSTMPredictor(
        sequence_length=90,  # Use 30 days of history
        prediction_horizon=1  # Predict next day
    )
    
    # Prepare data
    train_data, val_data, test_data = predictor.prepare_data(
        df, 
        target_column='returns',
        test_size=0.2,
        validation_size=0.1
    )
    
    # Train model
    history = predictor.train(
        train_data=train_data,
        val_data=val_data,
        epochs=200,
        batch_size=32,
        learning_rate=0.0001,
        patience=25,
        model_type='advanced' # which model
    )
    
    # Evaluate model
    metrics = predictor.evaluate(test_data, plot_results=True)
    
    # Save model
    predictor.model.save('final_lstm_model.h5')
    print("Model saved as 'final_lstm_model.h5'")
    
    return predictor, metrics

# Cross-validation for time series
def time_series_cross_validation(df, n_splits=5):
    """
    Perform time series cross-validation
    """
    predictor = StockLSTMPredictor(sequence_length=30)
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(df)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        # Split data
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]
        
        # Prepare data for this fold
        train_data, _, val_data = predictor.prepare_data(
            pd.concat([train_df, val_df]),
            test_size=len(val_df)/len(pd.concat([train_df, val_df])),
            validation_size=0
        )
        
        # Train model
        predictor.train(
            train_data, val_data,
            epochs=50, batch_size=32,
            patience=10, model_type='stacked'
        )
        
        # Evaluate
        metrics = predictor.evaluate(val_data, plot_results=False)
        cv_scores.append(metrics)
    
    # Average scores
    avg_metrics = {k: np.mean([score[k] for score in cv_scores]) 
                  for k in cv_scores[0].keys()}
    
    print("Cross-validation results:")
    for metric, value in avg_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    return cv_scores, avg_metrics

if __name__ == "__main__":
    # Run main training
    predictor, metrics = main()
    
    # Optional: Run cross-validation
    # cv_scores, avg_metrics = time_series_cross_validation(df)
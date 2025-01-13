import krakenex
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler  # Changed from MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import time
from typing import Tuple, Optional
from datetime import datetime, timedelta
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KrakenDataFetcher:
    def __init__(self, api_key: Optional[str] = None):
        self.api = krakenex.API()
        if api_key:
            self.api.key = api_key

    def fetch_data(self, days: int = 30) -> pd.DataFrame:
        url = "https://api.kraken.com/0/public/OHLC"
        params = {
            'pair': 'XXBTZUSD',
            'interval': 60,
            'since': int(time.time()) - 60 * 60 * 24 * days
        }

        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()

            if data.get('error'):
                logger.error(f"API Error: {data['error']}")
                return pd.DataFrame()

            ohlc = data['result']['XXBTZUSD']
            df = pd.DataFrame(
                ohlc,
                columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
            )

            df['time'] = pd.to_datetime(df['time'], unit='s')
            df['close'] = pd.to_numeric(df['close'], errors='coerce')

            # Calculate technical indicators with error handling
            df['SMA_20'] = self._safe_rolling_mean(df['close'], 20)
            df['RSI'] = self._safe_rsi(df['close'])

            # Remove any remaining NaN values
            df.dropna(inplace=True)
            return df[['time', 'close', 'SMA_20', 'RSI']]

        except requests.exceptions.RequestException as e:
            logger.error(f"Error fetching data from Kraken: {e}")
            return pd.DataFrame()

    @staticmethod
    def _safe_rolling_mean(series: pd.Series, window: int) -> pd.Series:
        """Calculate rolling mean with handling of NaN values"""
        result = series.rolling(window=window, min_periods=1).mean()
        return result.bfill()  # Changed from fillna(method='bfill')

    @staticmethod
    def _safe_rsi(prices: pd.Series, periods: int = 14) -> pd.Series:
        """Calculate RSI with proper error handling"""
        delta = prices.diff()
        delta = delta.fillna(0)  # Fill NA in first row

        # Make sure we don't divide by zero
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=periods, min_periods=1).mean()
        avg_loss = loss.rolling(window=periods, min_periods=1).mean()

        # Avoid division by zero
        avg_loss = avg_loss.replace(0, np.finfo(float).eps)

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50)  # Fill any remaining NaN with neutral RSI


class BitcoinCNN(nn.Module):
    def __init__(self, sequence_length: int):
        super(BitcoinCNN, self).__init__()

        # Reduced model complexity to prevent overfitting
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)

        self.batch_norm1 = nn.BatchNorm1d(16)
        self.batch_norm2 = nn.BatchNorm1d(32)

        self.dropout = nn.Dropout(0.1)  # Reduced dropout rate
        self.fc1 = nn.Linear(32 * sequence_length, 64)
        self.fc2 = nn.Linear(64, 1)
        self.relu = nn.ReLU()  # Changed back to ReLU for stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Add small epsilon to prevent numerical instability
        x = x + 1e-8

        x = self.relu(self.batch_norm1(self.conv1(x)))
        x = self.dropout(x)
        x = self.relu(self.batch_norm2(self.conv2(x)))
        x = self.dropout(x)

        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DataProcessor:
    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()  # Changed to StandardScaler for better stability

    def prepare_data(self, df: pd.DataFrame, train_split: float = 0.8) -> Tuple[torch.Tensor, ...]:
        features = ['close', 'SMA_20', 'RSI']

        # Ensure data is properly formatted and contains no infinities
        for feature in features:
            df[feature] = pd.to_numeric(df[feature], errors='coerce')
            df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
            df[feature] = df[feature].ffill()  # Changed from fillna(method='ffill')
            df[feature] = df[feature].bfill()

            # Scale the features
        scaled_data = self.scaler.fit_transform(df[features])

        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:i + self.sequence_length])
            y.append(scaled_data[i + self.sequence_length, 0])

        X, y = np.array(X), np.array(y)

        # Ensure no NaN values
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("NaN values found in processed data")

        # Train-test split
        train_size = int(len(X) * train_split)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # Convert to PyTorch tensors
        X_train = torch.tensor(X_train, dtype=torch.float32).transpose(1, 2)
        y_train = torch.tensor(y_train, dtype=torch.float32)
        X_test = torch.tensor(X_test, dtype=torch.float32).transpose(1, 2)
        y_test = torch.tensor(y_test, dtype=torch.float32)

        return X_train, y_train, X_test, y_test


# [Previous imports and initial classes remain the same until ModelTrainer]

class ModelTrainer:
    def __init__(self, model: nn.Module, criterion: nn.Module, optimizer: torch.optim.Optimizer):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def calculate_baseline_prediction(self, X: torch.Tensor) -> torch.Tensor:
        """Calculate baseline prediction using last value from sequence"""
        return X[:, 0, -1]  # Using last close price

    def train(self, X_train: torch.Tensor, y_train: torch.Tensor,
              X_test: torch.Tensor, y_test: torch.Tensor,
              batch_size: int = 32, epochs: int = 20) -> dict:
        history = {
            'train_loss': [],
            'val_loss': [],
            'baseline_loss': []
        }

        # Early stopping parameters
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        best_model_state = None

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0
            batch_count = 0

            # Training loop
            for i in range(0, len(X_train), batch_size):
                batch_end = min(i + batch_size, len(X_train))
                X_batch = X_train[i:batch_end].to(self.device)
                y_batch = y_train[i:batch_end].to(self.device)

                self.optimizer.zero_grad()

                try:
                    outputs = self.model(X_batch).squeeze()
                    loss = self.criterion(outputs, y_batch)

                    if torch.isnan(loss):
                        logger.error(f"NaN loss detected at epoch {epoch + 1}")
                        continue

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()

                    train_loss += loss.item()
                    batch_count += 1

                except RuntimeError as e:
                    logger.error(f"Error during training: {e}")
                    continue

            # Validation
            self.model.eval()
            with torch.no_grad():
                X_test_device = X_test.to(self.device)
                y_test_device = y_test.to(self.device)

                # Model validation loss
                val_outputs = self.model(X_test_device).squeeze()
                val_loss = self.criterion(val_outputs, y_test_device).item()

                # Baseline validation loss
                baseline_pred = self.calculate_baseline_prediction(X_test_device)
                baseline_loss = self.criterion(baseline_pred, y_test_device).item()

            # Calculate average losses
            avg_train_loss = train_loss / batch_count if batch_count > 0 else float('inf')

            # Store losses
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(val_loss)
            history['baseline_loss'].append(baseline_loss)

            logger.info(f"Epoch {epoch + 1}/{epochs}, "
                        f"Train Loss: {avg_train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}, "
                        f"Baseline Loss: {baseline_loss:.6f}")

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict()
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        return history


def plot_results(history: dict, y_test_rescaled: np.ndarray,
                 predictions_rescaled: np.ndarray,
                 last_hour_rescaled: np.ndarray) -> None:
    """Plot training history and predictions"""
    plt.figure(figsize=(15, 12))

    # Loss curves
    plt.subplot(3, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    if 'baseline_loss' in history:
        plt.plot(history['baseline_loss'], label='Baseline Loss', color='green')
    plt.title('Model Loss During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Price predictions
    plt.subplot(3, 1, 2)
    plt.plot(y_test_rescaled, label='Actual Prices', color='blue')
    plt.plot(predictions_rescaled, label='Model Predictions', color='red')
    plt.plot(last_hour_rescaled, label='Last Hour Prediction', color='green', alpha=0.5)
    plt.title('Bitcoin Price Predictions Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)

    # Prediction errors
    plt.subplot(3, 1, 3)
    model_errors = predictions_rescaled - y_test_rescaled
    baseline_errors = last_hour_rescaled - y_test_rescaled
    plt.plot(model_errors, label='Model Error', color='red', alpha=0.7)
    plt.plot(baseline_errors, label='Baseline Error', color='green', alpha=0.7)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Prediction Errors Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Error (USD)')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def print_metrics(y_test_rescaled: np.ndarray,
                  predictions_rescaled: np.ndarray,
                  last_hour_rescaled: np.ndarray) -> None:
    """Print detailed metrics for model and baseline predictions"""
    # Calculate metrics
    model_mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    baseline_mse = mean_squared_error(y_test_rescaled, last_hour_rescaled)
    model_r2 = r2_score(y_test_rescaled, predictions_rescaled)
    baseline_r2 = r2_score(y_test_rescaled, last_hour_rescaled)

    # Calculate errors
    model_errors = predictions_rescaled - y_test_rescaled
    baseline_errors = last_hour_rescaled - y_test_rescaled

    print("\nPrediction Statistics:")
    print(f"Model MSE: ${model_mse:.2f}")
    print(f"Baseline MSE: ${baseline_mse:.2f}")
    print(f"Model R² Score: {model_r2:.4f}")
    print(f"Baseline R² Score: {baseline_r2:.4f}")
    print(f"Model Mean Absolute Error: ${np.mean(np.abs(model_errors)):.2f}")
    print(f"Baseline Mean Absolute Error: ${np.mean(np.abs(baseline_errors)):.2f}")
    print(f"Model Standard Deviation of Error: ${np.std(model_errors):.2f}")
    print(f"Baseline Standard Deviation of Error: ${np.std(baseline_errors):.2f}")

    improvement = ((baseline_mse - model_mse) / baseline_mse) * 100
    print(f"\nModel Improvement over Baseline: {improvement:.2f}%")


def main():
    try:
        SEQUENCE_LENGTH = 48
        fetcher = KrakenDataFetcher()
        processor = DataProcessor(SEQUENCE_LENGTH)

        # Fetch and process data
        df = fetcher.fetch_data(days=180)
        if df.empty:
            logger.error("Failed to fetch data. Exiting.")
            return

        logger.info(f"Data retrieved: {len(df)} rows")
        logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")

        X_train, y_train, X_test, y_test = processor.prepare_data(df)

        # Model initialization
        model = BitcoinCNN(SEQUENCE_LENGTH)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)

        # Training
        trainer = ModelTrainer(model, criterion, optimizer)
        history = trainer.train(X_train, y_train, X_test, y_test)

        # Evaluation
        trainer.model.eval()
        with torch.no_grad():
            predictions = trainer.model(X_test.to(trainer.device)).cpu().squeeze().numpy()
            last_hour_predictions = trainer.calculate_baseline_prediction(X_test).cpu().numpy()
            y_test_numpy = y_test.numpy()

            # Prepare for inverse transform
            pred_3d = np.zeros((len(predictions), 3))
            pred_3d[:, 0] = predictions

            last_hour_3d = np.zeros((len(last_hour_predictions), 3))
            last_hour_3d[:, 0] = last_hour_predictions

            y_test_3d = np.zeros((len(y_test_numpy), 3))
            y_test_3d[:, 0] = y_test_numpy

            # Inverse transform
            predictions_rescaled = processor.scaler.inverse_transform(pred_3d)[:, 0]
            last_hour_rescaled = processor.scaler.inverse_transform(last_hour_3d)[:, 0]
            y_test_rescaled = processor.scaler.inverse_transform(y_test_3d)[:, 0]

        # Visualize and print results
        plot_results(history, y_test_rescaled, predictions_rescaled, last_hour_rescaled)
        print_metrics(y_test_rescaled, predictions_rescaled, last_hour_rescaled)

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
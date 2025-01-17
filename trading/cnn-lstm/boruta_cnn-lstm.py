import krakenex
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
import logging
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_bitcoin_data(days=30):
    """Fetch Bitcoin price data with all required features"""
    try:
        # Download Bitcoin data
        logger.info("Downloading Bitcoin data from Yahoo Finance...")
        df = yf.download('BTC-USD', period='1mo', interval='1h', progress=False)

        if df.empty:
            logger.error("No data retrieved")
            return pd.DataFrame()

        # Reset index and rename columns
        df = df.reset_index()

        # Process and rename all required columns
        df = df.rename(columns={
            'Datetime': 'time',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        })

        # Convert all price columns to numeric
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Calculate technical indicators
        df['SMA'] = df['close'].rolling(window=20).mean()
        df['RSI'] = calculate_rsi(df['close'])
        df['ma_50'] = df['close'].rolling(window=50).mean()
        df['volatility'] = calculate_volatility(df['close'])

        # Clean data
        df = df.dropna()

        logger.info(f"Data shape after processing: {df.shape}")
        logger.info(f"Available columns: {df.columns.tolist()}")

        return df

    except Exception as e:
        logger.error(f"Error in data fetching: {str(e)}")
        return pd.DataFrame()


def calculate_volatility(prices, window=20):
    """Calculate price volatility"""
    returns = np.log(prices / prices.shift(1))
    return returns.rolling(window=window).std() * np.sqrt(window)


def get_important_features():
    """Get important features using Boruta"""
    logger.info("Starting Boruta feature selection...")

    # Fetch data with all features
    df = fetch_bitcoin_data(days=30)

    if df.empty:
        raise ValueError("Failed to fetch data")

    # Verify columns
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'SMA', 'RSI', 'ma_50', 'volatility']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Setup Boruta
    X = df[required_columns]
    y = df['close']

    # Scale features for Boruta
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run Boruta
    rf = RandomForestRegressor(n_jobs=-1, random_state=42)
    boruta = BorutaPy(estimator=rf, n_estimators='auto', random_state=42)

    try:
        boruta.fit(X_scaled, y.values)
        selected_features = X.columns[boruta.support_].tolist()
        logger.info(f"Boruta selected features: {selected_features}")

        # Always include 'close' in selected features if not already selected
        if 'close' not in selected_features:
            selected_features.append('close')

        return selected_features, df

    except Exception as e:
        logger.error(f"Error in Boruta feature selection: {str(e)}")
        raise



def calculate_rsi(prices, periods=14):
    """Calculate RSI indicator"""
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    # Calculate average gains and losses
    avg_gains = gains.rolling(window=periods).mean()
    avg_losses = losses.rolling(window=periods).mean()

    # Handle division by zero
    avg_losses = avg_losses.replace(0, np.finfo(float).eps)

    rs = avg_gains / avg_losses
    rsi = 100 - (100 / (1 + rs))

    # Use bfill instead of fillna
    return rsi.bfill()



class PrepareBorutaCNNLSTMData:
    def __init__(self, sequence_length=48, train_split=0.8):
        self.sequence_length = sequence_length
        self.train_split = train_split
        self.price_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()

    def prepare_data(self, df, selected_features):
        """Prepare data using only Boruta-selected features"""
        # Scale price (target) separately
        prices = df[['close']].values
        scaled_prices = self.price_scaler.fit_transform(prices)

        # Scale selected features
        features = df[selected_features].values
        scaled_features = self.feature_scaler.fit_transform(features)

        # Combine scaled data
        scaled_data = np.hstack((scaled_prices, scaled_features))

        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - self.sequence_length):
            X.append(scaled_data[i:(i + self.sequence_length)])
            y.append(scaled_prices[i + self.sequence_length])

        X, y = np.array(X), np.array(y)

        # Train-test split
        train_size = int(len(X) * self.train_split)
        X_train = torch.FloatTensor(X[:train_size]).transpose(1, 2)
        y_train = torch.FloatTensor(y[:train_size])
        X_test = torch.FloatTensor(X[train_size:]).transpose(1, 2)
        y_test = torch.FloatTensor(y[train_size:])

        return X_train, y_train, X_test, y_test




def plot_results(history, y_true, y_pred, last_value_pred):
    """Plot training history and predictions"""
    plt.figure(figsize=(15, 10))

    # Loss curves
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    plt.grid(True)

    # Predictions
    plt.subplot(2, 1, 2)
    plt.plot(y_true, label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.plot(last_value_pred, label='Last Value', alpha=0.5)
    plt.title('Bitcoin Price Predictions')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()



def main():
    # Get Boruta-selected features
    selected_features, df = get_important_features()
    n_features = len(selected_features) + 1  # +1 for close price

    # Prepare data with selected features
    data_processor = PrepareBorutaCNNLSTMData()
    X_train, y_train, X_test, y_test = data_processor.prepare_data(df, selected_features)

    # Initialize model with correct number of features
    model = PrepareBorutaCNNLSTMData(n_features=n_features)

    # Train model
    model, history = train_model(model, X_train, y_train, X_test, y_test)

    # Make predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.to(device)).cpu().numpy()
        last_value = X_test[:, 0, -1].numpy()

        # Inverse transform predictions
        predictions_rescaled = data_processor.price_scaler.inverse_transform(predictions).flatten()
        last_value_rescaled = data_processor.price_scaler.inverse_transform(last_value.reshape(-1, 1)).flatten()
        y_true = data_processor.price_scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

    # Print metrics and plot results
    print_metrics(y_true, predictions_rescaled, last_value_rescaled)
    plot_results(history, y_true, predictions_rescaled, last_value_rescaled)


if __name__ == "__main__":
    main()

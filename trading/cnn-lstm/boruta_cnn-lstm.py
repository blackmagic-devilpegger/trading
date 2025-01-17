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
    """Fetch Bitcoin price data from Kraken with all required features"""
    url = "https://api.kraken.com/0/public/OHLC"
    params = {
        'pair': 'XXBTZUSD',
        'interval': 60,
        'since': int(time.time()) - 60 * 60 * 24 * days
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()['result']['XXBTZUSD']

        df = pd.DataFrame(data, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])

        # Convert columns to numeric
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col])

        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Calculate technical indicators
        # SMA 20
        df['SMA'] = df['close'].rolling(window=20).mean().bfill()

        # SMA 50
        df['ma_50'] = df['close'].rolling(window=50).mean().bfill()

        # RSI
        df['RSI'] = calculate_rsi(df['close'])

        # Volatility
        df['volatility'] = calculate_volatility(df['close'])

        # Select and return required columns
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume', 'SMA', 'RSI', 'ma_50', 'volatility']
        return df[required_columns]

    except Exception as e:
        logger.error(f"Error fetching data: {e}")
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

    # Drop the 'time' column as it's not a feature
    df = df.drop('time', axis=1)

    # Handle any NaN values using bfill() and ffill()
    df = df.bfill().ffill()

    # Verify columns
    required_columns = ['open', 'high', 'low', 'close', 'volume', 'SMA', 'RSI', 'ma_50', 'volatility']
    missing_columns = [col for col in required_columns if col not in df.columns]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Setup Boruta
    X = df.drop('close', axis=1)  # Remove target variable from features
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



class BitcoinCNNLSTM(nn.Module):
    def __init__(self, n_features):
        super(BitcoinCNNLSTM, self).__init__()

        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1)
        )

        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )

        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


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

        # Create sequences
        X, y = [], []
        for i in range(len(df) - self.sequence_length):
            # Get the sequence for both prices and features
            feature_seq = []
            for j in range(i, i + self.sequence_length):
                # Combine price and features for this timestep
                combined = np.hstack((scaled_prices[j], scaled_features[j]))
                feature_seq.append(combined)

            X.append(feature_seq)
            y.append(scaled_prices[i + self.sequence_length])

        X = np.array(X)
        y = np.array(y)

        # Train-test split
        train_size = int(len(X) * self.train_split)
        X_train = torch.FloatTensor(X[:train_size]).transpose(1, 2)
        y_train = torch.FloatTensor(y[:train_size])
        X_test = torch.FloatTensor(X[train_size:]).transpose(1, 2)
        y_test = torch.FloatTensor(y[train_size:])

        return X_train, y_train, X_test, y_test

def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    """Train the model with improved stability"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        batch_count = 0

        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size].to(device)
            batch_y = y_train[i:i + batch_size].to(device)

            optimizer.zero_grad()

            try:
                outputs = model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)

                if not torch.isnan(loss):
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                    batch_count += 1

            except RuntimeError as e:
                logger.error(f"Error in batch: {e}")
                continue

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test.to(device)).squeeze()
            val_loss = criterion(val_outputs, y_test.to(device)).item()

        if batch_count > 0:
            train_loss = train_loss / batch_count
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)

            scheduler.step(val_loss)

            logger.info(f"Epoch {epoch + 1}/{epochs}, "
                       f"Train Loss: {train_loss:.6f}, "
                       f"Val Loss: {val_loss:.6f}, "
                       f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    return model, history


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

    logger.info(f"Training model with {n_features} features: {selected_features}")

    # Prepare data with selected features
    data_processor = PrepareBorutaCNNLSTMData()
    X_train, y_train, X_test, y_test = data_processor.prepare_data(df, selected_features)

    # Initialize model with correct number of features
    model = BitcoinCNNLSTM(n_features=n_features)  # Changed from PrepareBorutaCNNLSTMData to BitcoinCNNLSTM

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
    mse = mean_squared_error(y_true, predictions_rescaled)
    r2 = r2_score(y_true, predictions_rescaled)
    print(f"\nMSE: ${mse:.2f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Mean Absolute Error: ${np.mean(np.abs(y_true - predictions_rescaled)):.2f}")

    plot_results(history, y_true, predictions_rescaled, last_value_rescaled)

if __name__ == "__main__":
    main()

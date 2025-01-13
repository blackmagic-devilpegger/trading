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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_bitcoin_data(days=30):
    """Fetch Bitcoin price data from Kraken"""
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
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['close'] = pd.to_numeric(df['close'])

        # Fixed deprecated fillna
        sma = df['close'].rolling(window=20).mean()
        df['SMA'] = sma.bfill()  # Using bfill() instead of fillna(method='bfill')
        df['RSI'] = calculate_rsi(df['close'])

        return df[['time', 'close', 'SMA', 'RSI']]
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()


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
    def __init__(self, n_features=3):
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

        # Fixed LSTM layer configuration
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=32,
            num_layers=2,  # Increased to 2 layers to match dropout expectation
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
        x = x + 1e-8
        x = self.cnn(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x


def train_model(model, X_train, y_train, X_test, y_test, epochs=20, batch_size=32):
    """Train the model with improved stability"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()

    # Use AdamW with weight decay for better stability
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    # Learning rate scheduler
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

                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
                    train_loss += loss.item()
                    batch_count += 1

            except RuntimeError as e:
                print(f"Error in batch: {e}")
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

            # Learning rate scheduling
            scheduler.step(val_loss)

            logger.info(f"Epoch {epoch + 1}/{epochs}, "
                        f"Train Loss: {train_loss:.6f}, "
                        f"Val Loss: {val_loss:.6f}, "
                        f"LR: {optimizer.param_groups[0]['lr']:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                break

    return model, history


def prepare_data(df, sequence_length=48, train_split=0.8):
    """Prepare data with separate scaling for price and other features"""
    # Create separate scalers
    price_scaler = StandardScaler()
    feature_scaler = StandardScaler()

    # Scale close price separately
    prices = df[['close']].values
    scaled_prices = price_scaler.fit_transform(prices)

    # Scale other features
    features = df[['SMA', 'RSI']].values
    scaled_features = feature_scaler.fit_transform(features)

    # Combine scaled data
    scaled_data = np.hstack((scaled_prices, scaled_features))

    # Create sequences
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:(i + sequence_length)])
        y.append(scaled_prices[i + sequence_length])

    X, y = np.array(X), np.array(y)

    # Train-test split
    train_size = int(len(X) * train_split)
    X_train = torch.FloatTensor(X[:train_size]).transpose(1, 2)
    y_train = torch.FloatTensor(y[:train_size])
    X_test = torch.FloatTensor(X[train_size:]).transpose(1, 2)
    y_test = torch.FloatTensor(y[train_size:])

    return X_train, y_train, X_test, y_test, price_scaler


def main():
    # Fetch and prepare data
    df = fetch_bitcoin_data(days=30)
    if df.empty:
        logger.error("Failed to fetch data")
        return

    # Print raw data stats
    print("\nRaw Data Statistics:")
    print(f"Price Range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
    print(f"Average Price: ${df['close'].mean():.2f}")

    X_train, y_train, X_test, y_test, price_scaler = prepare_data(df)

    # Initialize and train model
    model = BitcoinCNNLSTM()
    model, history = train_model(model, X_train, y_train, X_test, y_test)

    # Make predictions
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    with torch.no_grad():
        predictions = model(X_test.to(device)).cpu().numpy()
        last_value = X_test[:, 0, -1].numpy()

        # Inverse transform predictions directly
        predictions_rescaled = price_scaler.inverse_transform(predictions).flatten()
        last_value_rescaled = price_scaler.inverse_transform(last_value.reshape(-1, 1)).flatten()
        y_true = price_scaler.inverse_transform(y_test.numpy().reshape(-1, 1)).flatten()

    # Print prediction samples
    print("\nSample Predictions:")
    for i in range(min(5, len(y_true))):
        print(f"Actual: ${y_true[i]:,.2f} | Predicted: ${predictions_rescaled[i]:,.2f} | "
              f"Error: ${abs(y_true[i] - predictions_rescaled[i]):,.2f}")

    # Calculate metrics
    mse = mean_squared_error(y_true, predictions_rescaled)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - predictions_rescaled))
    r2 = r2_score(y_true, predictions_rescaled)

    print("\nModel Performance:")
    print(f"RMSE: ${rmse:,.2f}")
    print(f"MAE: ${mae:,.2f}")
    print(f"R² Score: {r2:.4f}")

    # Plot results
    plot_results(history, y_true, predictions_rescaled, last_value_rescaled)


def plot_results(history, y_true, y_pred, last_value_pred):
    """Enhanced plotting with proper price formatting"""
    plt.figure(figsize=(15, 10))

    # Loss curves
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss', color='blue')
    plt.plot(history['val_loss'], label='Validation Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # Price predictions
    plt.subplot(2, 1, 2)
    time_steps = range(len(y_true))
    plt.plot(time_steps, y_true, label='Actual Price', color='blue', linewidth=2)
    plt.plot(time_steps, y_pred, label='Predicted Price', color='red', alpha=0.8)
    plt.plot(time_steps, last_value_pred, label='Last Known Price', color='green', alpha=0.5)

    plt.title('Bitcoin Price Predictions')
    plt.xlabel('Time Steps')
    plt.ylabel('Price (USD)')

    # Format y-axis as currency
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['${:,.0f}'.format(x) for x in current_values])

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Additional error plot
    plt.figure(figsize=(15, 5))
    prediction_error = y_pred - y_true
    plt.plot(time_steps, prediction_error, color='red', label='Prediction Error')
    plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    plt.title('Prediction Error Over Time')
    plt.xlabel('Time Steps')
    plt.ylabel('Error (USD)')
    plt.legend()
    plt.grid(True)

    # Format y-axis as currency
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['${:,.0f}'.format(x) for x in current_values])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

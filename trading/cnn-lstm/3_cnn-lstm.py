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
import yfinance as yf
import pandas as pd
import numpy as np
import logging

# Set up logging


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

import yfinance as yf
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def main():
    """Main function to fetch data and train the model"""
    try:
        # 1. Fetch and prepare data
        logger.info("Downloading Bitcoin data from Yahoo Finance...")
        df = yf.download('BTC-USD', period='1mo', interval='1h', progress=False)

        if df.empty:
            logger.error("No data retrieved from Yahoo Finance")
            return

        # Handle MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Reset index to get datetime as column
        df = df.reset_index()

        # 2. Process data
        logger.info("Processing data and calculating indicators...")
        processed_df = pd.DataFrame()
        processed_df['time'] = df['Datetime']
        processed_df['close'] = df['Close']

        # Calculate SMA
        processed_df['SMA'] = processed_df['close'].rolling(window=20, min_periods=1).mean()

        # Calculate RSI
        delta = processed_df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()
        avg_loss = avg_loss.replace(0, np.finfo(float).eps)

        rs = avg_gain / avg_loss
        processed_df['RSI'] = 100 - (100 / (1 + rs))

        # Clean data
        processed_df = processed_df.dropna()

        # 3. Print data information
        logger.info(f"Successfully processed {len(processed_df)} rows of data")
        logger.info(f"Date range: {processed_df['time'].min()} to {processed_df['time'].max()}")
        logger.info(f"Price range: ${processed_df['close'].min():.2f} to ${processed_df['close'].max():.2f}")

        print("\nData Statistics:")
        print(f"Total rows: {len(processed_df)}")
        print(f"Time range: {processed_df['time'].min()} to {processed_df['time'].max()}")
        print(f"Price range: ${processed_df['close'].min():.2f} to ${processed_df['close'].max():.2f}")
        print(f"RSI range: {processed_df['RSI'].min():.2f} to {processed_df['RSI'].max():.2f}")

        print("\nFirst 5 rows:")
        print(processed_df.head())

        print("\nData Summary:")
        print(processed_df.describe())

        # 4. Prepare data for model
        # [Your model preparation code would go here]

        return processed_df

    except Exception as e:
        logger.error(f"Error in data processing: {str(e)}")
        return None


if __name__ == "__main__":
    df = main()




def fetch_bitcoin_data(days=30):
    """
    Fetch historical Bitcoin price data from Yahoo Finance.

    Args:
        days (int): Number of days of historical data to fetch (ignored, using '1mo' period)

    Returns:
        pandas.DataFrame: DataFrame with time, close price, SMA, and RSI
    """
    try:
        # Download Bitcoin data with correct period parameter
        df = yf.download(
            tickers='BTC-USD',
            period='1mo',  # Changed from '30d' to '1mo'
            interval='1h',
            progress=False
        )

        if df.empty:
            logger.error("No data retrieved from Yahoo Finance")
            return pd.DataFrame()

        # Process the data
        df.reset_index(inplace=True)

        # Ensure correct column names
        if 'Date' in df.columns:
            df.rename(columns={'Date': 'time'}, inplace=True)
        elif 'Datetime' in df.columns:
            df.rename(columns={'Datetime': 'time'}, inplace=True)

        df.rename(columns={'Close': 'close'}, inplace=True)

        # Calculate SMA
        df['SMA'] = df['close'].rolling(
            window=20,
            min_periods=1
        ).mean()

        # Calculate RSI
        delta = df['close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        avg_gain = gain.rolling(window=14, min_periods=1).mean()
        avg_loss = loss.rolling(window=14, min_periods=1).mean()

        rs = avg_gain / (avg_loss + np.finfo(float).eps)
        df['RSI'] = 100 - (100 / (1 + rs))

        # Clean data
        df.dropna(subset=['close', 'SMA', 'RSI'], inplace=True)

        # Log successful fetch
        logger.info(f"Successfully fetched {len(df)} rows of Bitcoin data")
        logger.info(f"Date range: {df['time'].min()} to {df['time'].max()}")
        logger.info(f"Price range: ${df['close'].min():.2f} to ${df['close'].max():.2f}")

        return df[['time', 'close', 'SMA', 'RSI']]

    except Exception as e:
        logger.error(f"Error fetching data: {str(e)}")
        return pd.DataFrame()


def test_data_fetching():
    """Test the data fetching function"""
    print("Fetching Bitcoin data...")
    df = fetch_bitcoin_data()

    if not df.empty:
        print("\nSample of fetched data:")
        print(df.head())
        print("\nData shape:", df.shape)
        print("\nColumn statistics:")
        print(df.describe())

        # Additional data validation
        print("\nData validation:")
        print(f"Number of rows: {len(df)}")
        print(f"Missing values:\n{df.isnull().sum()}")
        print(f"\nPrice range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")
        print(f"RSI range: {df['RSI'].min():.2f} - {df['RSI'].max():.2f}")
    else:
        print("Failed to fetch data")


if __name__ == "__main__":
    test_data_fetching()



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
'''

if __name__ == "__main__":
    main() '''

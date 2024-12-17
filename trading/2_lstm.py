import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


def fetch_bitcoin_historical_data():
    """
    Fetch Bitcoin historical price data from a reliable public API

    Returns:
    pandas.DataFrame with historical price data
    """
    # Use CoinGecko API as an alternative data source
    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart?vs_currency=usd&days=3650"

    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Extract price data
        prices = data['prices']
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])

        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
        df = df[['time', 'price']]
        df.columns = ['time', 'close']

        return df

    except Exception as e:
        print(f"Data retrieval error: {e}")
        return pd.DataFrame()


# Fetch historical Bitcoin data
df = fetch_bitcoin_historical_data()

# Verify data retrieval
if df.empty:
    print("No historical data retrieved. Check API connection.")
    exit()

# Print data overview
print(f"Total historical data points: {len(df)}")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")


def prepare_lstm_data(data, train_start, train_end, test_start, test_end, sequence_length=48):
    """
    Prepare training and testing data for LSTM
    """
    # Filter data for training and testing
    df_train = data[(data['time'] >= train_start) & (data['time'] <= train_end)]
    df_test = data[(data['time'] >= test_start) & (data['time'] <= test_end)]

    print(f"Training data range: {df_train['time'].min()} to {df_train['time'].max()}")
    print(f"Training data points: {len(df_train)}")
    print(f"Testing data range: {df_test['time'].min()} to {df_test['time'].max()}")
    print(f"Testing data points: {len(df_test)}")

    def create_sequences(data, sequence_length):
        # Normalize the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data[['close']])

        # Create sequences
        X, y = [], []
        for i in range(len(scaled_data) - sequence_length):
            X.append(scaled_data[i:i + sequence_length])
            y.append(scaled_data[i + sequence_length])

        return (np.array(X).reshape(-1, sequence_length, 1),
                np.array(y),
                scaler)

    # Create sequences for train and test
    X_train, y_train, train_scaler = create_sequences(df_train, sequence_length)
    X_test, y_test, test_scaler = create_sequences(df_test, sequence_length)

    return (X_train, y_train, train_scaler,
            X_test, y_test, test_scaler)


# Define date ranges
train_start = df['time'].min()
train_end = df['time'].max() - pd.DateOffset(years=1)
test_start = train_end + pd.DateOffset(days=1)
test_end = df['time'].max()

# Prepare LSTM data
try:
    X_train, y_train, train_scaler, X_test, y_test, test_scaler = prepare_lstm_data(
        df, train_start, train_end, test_start, test_end
    )
except ValueError as e:
    print(f"Data preparation error: {e}")
    exit()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


# Rest of your existing LSTM model code remains the same

# Rest of the code remains the same...
# 3. Define the LSTM model
class BitcoinLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(BitcoinLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return output

input_size = 1
hidden_size = 256
num_layers = 2
model = BitcoinLSTM(input_size, hidden_size, num_layers)

# 4. Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Train the model with training and validation loss
train_loss_values = []
val_loss_values = []
num_epochs = 20 # more - slightly better
batch_size = 64

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0

    for i in range(0, len(X_train), batch_size):
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs.squeeze(), y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    train_loss = epoch_loss / len(X_train)
    train_loss_values.append(train_loss)

    # Validation
    model.eval()
    with torch.no_grad():
        all_predictions = []
        all_actual_prices = []

        for i in range(len(X_test_tensor)):
            val_output = model(X_test_tensor[i].unsqueeze(0)).squeeze()

            prediction = val_output.numpy()
            actual_price = y_test_tensor[i].numpy()

            all_predictions.append(prediction)
            all_actual_prices.append(actual_price)

        predictions = np.array(all_predictions)
        y_test_numpy = np.array(all_actual_prices)

        # Use test_scaler for rescaling
        y_test_rescaled = test_scaler.inverse_transform(y_test_numpy.reshape(-1, 1)).flatten()
        predictions_rescaled = test_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

    # Calculate performance metrics
    mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    print(f"Mean Squared Error (MSE) on Test Data: {mse:.4f}")

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}")

# 6. Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), train_loss_values, label="Training Loss")
plt.plot(range(num_epochs), val_loss_values, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.title("Training and Validation Loss")
plt.show()

# 7. Validate the model with a forward pass for each predicted price
model.eval()
with torch.no_grad():
    # Initialize lists to store predictions and actual values
    all_predictions = []
    all_actual_prices = []

    # Iterate through validation data
    for i in range(len(X_val)):
        # Perform a forward pass for each individual sequence
        val_output = model(X_val[i].unsqueeze(0)).squeeze()

        # Store predictions and actual prices
        prediction = val_output.numpy()
        actual_price = y_val[i].numpy()

        all_predictions.append(prediction)
        all_actual_prices.append(actual_price)

    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    y_val_numpy = np.array(all_actual_prices)

    # Rescale predictions and actual prices
    y_val_rescaled = scaler.inverse_transform(y_val_numpy.reshape(-1, 1)).flatten()
    predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

# Calculate the Mean Squared Error (MSE) between actual and predicted prices
mse = mean_squared_error(y_val_rescaled, predictions_rescaled)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Calculate deviations
last_hour_prices = []
for i in range(len(y_val_rescaled)):
    # For each prediction point, use the previous point's price as the "last hour" price
    last_hour_price = y_val_rescaled[i-1] if i > 0 else y_val_rescaled[0]
    last_hour_prices.append(last_hour_price)

# Deviations of predicted prices from actual prices
predicted_deviations = predictions_rescaled - y_val_rescaled

# Deviations of average price from actual prices
last_hour_deviations = np.array(last_hour_prices) - y_val_rescaled

# Mean Absolute Deviation for predictions
mad_predictions = np.mean(np.abs(predicted_deviations))
print(f"Mean Absolute Deviation (MAD) of Predictions: {mad_predictions:.4f} USD")

# Mean Absolute Deviation for average price
mad_last_hour = np.mean(np.abs(last_hour_deviations))
print(f"Mean Absolute Deviation (MAD) of Last Hour Price: {mad_last_hour:.4f} USD")

# Plot actual vs. predicted prices and deviations
plt.figure(figsize=(12, 6))
plt.plot(y_val_rescaled, label="Actual Prices", color='blue', alpha=0.7)
plt.plot(predictions_rescaled, label="Predicted Prices", color='red', alpha=0.7)
plt.plot(last_hour_prices, color='green', linestyle='--', label="Last Hour Price")

plt.title("Bitcoin Price Prediction with Deviations")
plt.xlabel("Time Steps")
plt.ylabel("Price (USD) / Deviations")
plt.legend()
plt.show()

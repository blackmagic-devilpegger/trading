import krakenex
import requests
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time

# Initialize Kraken API
api = krakenex.API()
api.key = 'w3if4ZjPEKdgCVsj7J/KVRgkSKhAhYBcJJrrp8gXTfrRdlylAVafK85F'

def fetch_kraken_data():
    url = "https://api.kraken.com/0/public/OHLC"
    params = {
        'pair': 'XXBTZUSD',  # Bitcoin (XBT) to USD
        'interval': 60,      # Time interval (60 minutes)
        'since': int(time.time()) - 60 * 60 * 24 * 30  # Data from the last 30 days (longest possible period)
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if len(data['error']) > 0:
            print(f"API Error: {data['error']}")
            return pd.DataFrame()

        # Extract OHLC data
        ohlc = data['result']['XXBTZUSD']
        df = pd.DataFrame(ohlc, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])

        # Convert timestamp to datetime
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['close'] = pd.to_numeric(df['close'], errors='coerce')

        # Drop rows with invalid (NaN) values in the 'close' column
        df.dropna(subset=['close'], inplace=True)
        return df[['time', 'close']]

    except Exception as e:
        print(f"Error fetching data from Kraken: {e}")
        return pd.DataFrame()

# Fetch data using Kraken API
df = fetch_kraken_data()

# Verify data retrieval
if df.empty:
    print("No data retrieved from Kraken API. Exiting.")
    exit()

# Print data overview
print(f"Data retrieved: {len(df)} rows")
print(f"Date range: {df['time'].min()} to {df['time'].max()}")

sequence_length=48
scaler = MinMaxScaler()

def prepare_lstm_data(data, train_start, train_end, test_start, test_end, sequence_length):
    # Filter data for training and testing
    df_train = data[(data['time'] >= train_start) & (data['time'] <= train_end)]
    df_test = data[(data['time'] >= test_start) & (data['time'] <= test_end)]

    if df_train.empty:
        raise ValueError("Training data is empty. Check the date ranges.")
    if df_test.empty:
        raise ValueError("Testing data is empty. Check the date ranges.")

    print(f"Training data range: {df_train['time'].min()} to {df_train['time'].max()}")
    print(f"Training data points: {len(df_train)}")
    print(f"Testing data range: {df_test['time'].min()} to {df_test['time'].max()}")
    print(f"Testing data points: {len(df_test)}")

    def create_sequences(data, sequence_length):
        # Normalize the data
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
    return (X_train, y_train, train_scaler, X_test, y_test, test_scaler)

# Adjust training and testing split
train_start = df['time'].min()
train_end = df['time'].max() - pd.DateOffset(weeks=3)  # Allocate more data for training
test_start = train_end + pd.DateOffset(hours=1)
test_end = df['time'].max()

print(f"Training Start: {train_start}, Training End: {train_end}")
print(f"Testing Start: {test_start}, Testing End: {test_end}")

# Debug: Check the filtered data
df_train_check = df[(df['time'] >= train_start) & (df['time'] <= train_end)]
df_test_check = df[(df['time'] >= test_start) & (df['time'] <= test_end)]

print(f"Training Data Points: {len(df_train_check)}")
print(f"Testing Data Points: {len(df_test_check)}")

if len(df_train_check) <= sequence_length:
    print(f"Error: Not enough data points for sequence preparation. At least {sequence_length + 1} required.")
    exit()

# Prepare LSTM data
try:
    X_train, y_train, train_scaler, X_test, y_test, test_scaler = prepare_lstm_data(
        df, train_start, train_end, test_start, test_end, sequence_length=48
    )
except ValueError as e:
    print(f"Data preparation error: {e}")
    exit()

if len(X_train) == 0:
    print("Error: Training data is empty after sequence preparation.")
    exit()

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

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
# Training loop with baseline comparison
train_loss_values = []
val_loss_values = []
baseline_loss_values = []  # New list to store last hour price losses
num_epochs = 20
batch_size = 64

for epoch in range(num_epochs):
    # Training phase
    model.train()
    epoch_loss = 0

    for i in range(0, len(X_train), batch_size):
        # Get the batch
        X_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size]

        # Convert to PyTorch tensors
        X_batch_tensor = torch.tensor(X_batch, dtype=torch.float32)
        y_batch_tensor = torch.tensor(y_batch, dtype=torch.float32)

        optimizer.zero_grad()  # Reset gradients
        outputs = model(X_batch_tensor)  # Forward pass
        loss = criterion(outputs.squeeze(), y_batch_tensor.squeeze())
        loss.backward()  # Backpropagation
        optimizer.step()  # Update model weights

        epoch_loss += loss.item()  # Accumulate epoch loss

    train_loss = epoch_loss / len(X_train)
    train_loss_values.append(train_loss)

    # Validation phase
    model.eval()
    val_loss = 0
    baseline_loss = 0  # Initialize baseline loss
    all_predictions = []
    all_actual_prices = []
    last_hour_predictions = []  # Store last hour predictions

    with torch.no_grad():
        for i in range(0, len(X_test_tensor), batch_size):
            # Get validation batch
            X_val_batch = X_test_tensor[i:i + batch_size]
            y_val_batch = y_test_tensor[i:i + batch_size]

            # Forward pass
            val_outputs = model(X_val_batch).squeeze()

            # For baseline, use the last value from each input sequence
            last_hour_vals = X_val_batch[:, -1, 0]  # Get last value from each sequence

            # Compute losses
            batch_loss = criterion(val_outputs, y_val_batch.squeeze()).item()
            baseline_batch_loss = criterion(last_hour_vals, y_val_batch.squeeze()).item()

            val_loss += batch_loss
            baseline_loss += baseline_batch_loss

            # Collect predictions and actual prices for rescaling
            all_predictions.extend(val_outputs.numpy())
            all_actual_prices.extend(y_val_batch.numpy())
            last_hour_predictions.extend(last_hour_vals.numpy())

    # Average validation and baseline losses
    val_loss /= (len(X_test_tensor) / batch_size)
    baseline_loss /= (len(X_test_tensor) / batch_size)

    val_loss_values.append(val_loss)
    baseline_loss_values.append(baseline_loss)

    # Rescale predictions and actual values
    predictions = np.array(all_predictions)
    y_test_numpy = np.array(all_actual_prices)
    last_hour_numpy = np.array(last_hour_predictions)

    y_test_rescaled = test_scaler.inverse_transform(y_test_numpy.reshape(-1, 1)).flatten()
    predictions_rescaled = test_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    last_hour_rescaled = test_scaler.inverse_transform(last_hour_numpy.reshape(-1, 1)).flatten()

    # Calculate performance metrics
    model_mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
    baseline_mse = mean_squared_error(y_test_rescaled, last_hour_rescaled)

    print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {train_loss:.6f}, Validation Loss: {val_loss:.6f}, "
          f"Last Hour Baseline Loss: {baseline_loss:.6f}, Model MSE: {model_mse:.2f}, Baseline MSE: {baseline_mse:.2f}")

# 6. Plot training, validation, and baseline losses
plt.figure(figsize=(12, 6))
plt.plot(range(num_epochs), train_loss_values, label="Training Loss", color='blue')
plt.plot(range(num_epochs), val_loss_values, label="Validation Loss", color='red')
plt.plot(range(num_epochs), baseline_loss_values, label="Last Hour Baseline Loss", color='green')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training, Validation, and Baseline Losses")
plt.legend()
plt.grid(True)
plt.show()

# Final evaluation metrics
print("\nFinal Model Evaluation:")
print(f"Model MSE: {model_mse:.2f}")
print(f"Baseline MSE: {baseline_mse:.2f}")
print(f"Model improvement over baseline: {((baseline_mse - model_mse) / baseline_mse * 100):.2f}%")

# Calculate and display Mean Absolute Error (MAE)
model_mae = np.mean(np.abs(predictions_rescaled - y_test_rescaled))
baseline_mae = np.mean(np.abs(last_hour_rescaled - y_test_rescaled))
print(f"Model MAE: {model_mae:.2f} USD")
print(f"Baseline MAE: {baseline_mae:.2f} USD")

# 7. Validate the model with a forward pass for each predicted price
model.eval()
with torch.no_grad():
    # Initialize lists to store predictions and actual values
    all_predictions = []
    all_actual_prices = []

    # Iterate through test data in batches
    for i in range(0, len(X_test_tensor), batch_size):
        # Get batch
        X_batch = X_test_tensor[i:i + batch_size]
        y_batch = y_test_tensor[i:i + batch_size]

        # Perform forward pass
        outputs = model(X_batch)

        # Store predictions and actual prices
        all_predictions.extend(outputs.squeeze().numpy())
        all_actual_prices.extend(y_batch.numpy())

    # Convert to numpy arrays
    predictions = np.array(all_predictions)
    y_test_numpy = np.array(all_actual_prices)

    # Rescale predictions and actual prices
    y_test_rescaled = test_scaler.inverse_transform(y_test_numpy.reshape(-1, 1)).flatten()
    predictions_rescaled = test_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()

# Calculate the Mean Squared Error (MSE) between actual and predicted prices
mse = mean_squared_error(y_test_rescaled, predictions_rescaled)
print(f"Mean Squared Error (MSE): {mse:.4f}")

# Calculate deviations
last_hour_prices = []
for i in range(len(y_test_rescaled)):
    # For each prediction point, use the previous point's price as the "last hour" price
    last_hour_price = y_test_rescaled[i - 1] if i > 0 else y_test_rescaled[0]
    last_hour_prices.append(last_hour_price)

# Deviations of predicted prices from actual prices
predicted_deviations = predictions_rescaled - y_test_rescaled
# Deviations of last hour price from actual prices
last_hour_deviations = np.array(last_hour_prices) - y_test_rescaled

# Mean Absolute Deviation for predictions
mad_predictions = np.mean(np.abs(predicted_deviations))
print(f"Mean Absolute Deviation (MAD) of Predictions: {mad_predictions:.4f} USD")
# Mean Absolute Deviation for last hour price
mad_last_hour = np.mean(np.abs(last_hour_deviations))
print(f"Mean Absolute Deviation (MAD) of Last Hour Price: {mad_last_hour:.4f} USD")

#8. Plot prices
plt.figure(figsize=(12, 6))
plt.plot(y_test_rescaled, label="Actual Prices", color='blue', alpha=0.7)
plt.plot(predictions_rescaled, label="Predicted Prices", color='red', alpha=0.7)
plt.plot(last_hour_prices, label="Last Hour Price", color='green', linestyle='--', alpha=0.7)
plt.title("Bitcoin Price Prediction Performance")
plt.xlabel("Time Steps")
plt.ylabel("Price (USD)")
plt.legend()
plt.grid(True)
plt.show()

# Plot prediction errors
plt.figure(figsize=(12, 6))
plt.plot(predicted_deviations, label="Prediction Error", color='red', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
plt.title("Prediction Errors Over Time")
plt.xlabel("Time Steps")
plt.ylabel("Error (USD)")
plt.legend()
plt.grid(True)
plt.show()
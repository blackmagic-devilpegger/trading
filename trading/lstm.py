import krakenex
import requests
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Initialize Kraken API
api = krakenex.API()
api.key = 'w3if4ZjPEKdgCVsj7J/KVRgkSKhAhYBcJJrrp8gXTfrRdlylAVafK85F'

# Fetch current ticker data
response = api.query_public('Ticker', {'pair': 'XXBTZUSD'})
print(response['result'])

# 1. Fetch historical Bitcoin data from Kraken API
url = "https://api.kraken.com/0/public/OHLC"
params = {
    'pair': 'XXBTZUSD',
    'interval': 60,  # Time interval (60 minutes)
    'since': int(time.time()) - 60 * 60 * 24 * 30  # Data from the last 30 days
}
response = requests.get(url, params=params)
data = response.json()

if len(data['error']) == 0:  # No errors
    ohlc = data['result']['XXBTZUSD']
    df = pd.DataFrame(
        ohlc, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
    )
    df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert time to datetime
else:
    print("Error:", data['error'])
    exit()

# Ensure the 'close' column is numeric
df['close'] = pd.to_numeric(df['close'], errors='coerce')

# Drop rows with invalid (NaN) values in the 'close' column
df.dropna(subset=['close'], inplace=True)

# Calculate the average Bitcoin price in the selected time period
average_price = df['close'].iloc[-1]  # Get the price of the last hour
print(f"Average Bitcoin price in the selected period: {average_price:.2f} USD")


# 2. Prepare data for LSTM
df = df[['time', 'close']]
df.set_index('time', inplace=True)

# Normalize the data
scaler = MinMaxScaler()
df['close_scaled'] = scaler.fit_transform(df[['close']])

# Create sequences for LSTM
sequence_length = 48  # higher - line
X, y = [], []
data = df['close_scaled'].values

for i in range(len(data) - sequence_length):
    X.append(data[i:i + sequence_length])
    y.append(data[i + sequence_length])

X = np.array(X).reshape(-1, sequence_length, 1)
y = np.array(y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split into training and validation sets
split_index = int(len(X_tensor) * 0.8)
X_train, X_val = X_tensor[:split_index], X_tensor[split_index:]
y_train, y_val = y_tensor[:split_index], y_tensor[split_index:]

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
    val_loss = 0
    with torch.no_grad():
        for i in range(0, len(X_val), batch_size):
            X_val_batch = X_val[i:i + batch_size]
            y_val_batch = y_val[i:i + batch_size]
            outputs = model(X_val_batch)
            val_loss += criterion(outputs.squeeze(), y_val_batch).item()

    val_loss /= len(X_val)
    val_loss_values.append(val_loss)

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

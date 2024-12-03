import krakenex

api = krakenex.API()
api.key = 'w3if4ZjPEKdgCVsj7J/KVRgkSKhAhYBcJJrrp8gXTfrRdlylAVafK85F'

# API fuction call
response = api.query_public('Ticker', {'pair': 'XXBTZUSD'})
print(response['result'])

import requests
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 1. Fetch historical Bitcoin data from Kraken API
url = "https://api.kraken.com/0/public/OHLC"
params = {
    'pair': 'XXBTZUSD',  # Bitcoin (XBT) to US Dollar (USD)
    'interval': 60,      # Time interval (e.g., 60 minutes)
    'since': int(time.time()) - 60 * 60 * 24 * 90  # Data from the last 90 days
}

response = requests.get(url, params=params)
data = response.json()

if len(data['error']) == 0:  # No errors
    ohlc = data['result']['XXBTZUSD']
    df = pd.DataFrame(
        ohlc, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
    )
    df['time'] = pd.to_datetime(df['time'], unit='s')  # Convert time to datetime
    print(df.head())
else:
    print("Error:", data['error'])
    exit()

# 2. Prepare data for LSTM
df = df[['time', 'close']]
df.set_index('time', inplace=True)

# Normalize the data
scaler = MinMaxScaler()
df['close_scaled'] = scaler.fit_transform(df[['close']])

# Create sequences for LSTM
sequence_length = 48  # 48 hours (2 days)
X, y = [], []
data = df['close_scaled'].values

for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length])  # Time window
    y.append(data[i+sequence_length])    # Next closing price

X = np.array(X).reshape(-1, sequence_length, 1)  # Reshape for LSTM (batch, seq_len, features)
y = np.array(y)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split into train and validation sets
# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Split into train and validation sets (last 20% as test data)
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
        out, (hidden, _) = self.lstm(x)
        output = self.fc(hidden[-1])
        return output

input_size = 1  # Only the 'close' feature
hidden_size = 64 # verdoppelung - 2 USD unterschied
num_layers = 2
model = BitcoinLSTM(input_size, hidden_size, num_layers)

# 4. Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. Train the model
loss_vals = []
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()

    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs.squeeze(), y_train)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    # Store the loss for visualization
    loss_vals.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 6. Plot training loss
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(range(num_epochs), loss_vals, label="Training Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss Over Epochs")
plt.legend()
plt.show()

# 7. Validate the model
model.eval()
with torch.no_grad():
    val_outputs = model(X_val)
    val_loss = criterion(val_outputs.squeeze(), y_val)
    print(f"Validation Loss: {val_loss.item():.4f}")



# 8. Make predictions
model.eval()
with torch.no_grad():
    predictions = model(X_val).squeeze()  # Vorhersagen für die Validierungsdaten
    predictions = predictions.numpy()  # Umwandlung in NumPy für Rückskalierung
    y_val_np = y_val.numpy()  # Zielwerte als NumPy-Array für Vergleich

# Rückskalierung der Vorhersagen
predictions_rescaled = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
y_val_rescaled = scaler.inverse_transform(y_val_np.reshape(-1, 1)).flatten()

# Berechnung des durchschnittlichen Abweichung (Mean Absolute Error)
mae = np.mean(np.abs(predictions_rescaled - y_val_rescaled))
print(f"Mean Absolute Error (MAE): {mae:.4f}")

# Visualisierung der Vorhersagen
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_val_rescaled, label="Actual Prices", color='blue', alpha=0.7)
plt.plot(predictions_rescaled, label="Predicted Prices", color='red', alpha=0.7)
plt.title("Bitcoin Price Prediction")
plt.xlabel("Time Steps")
plt.ylabel("Price (USD)")
plt.legend()
plt.show()



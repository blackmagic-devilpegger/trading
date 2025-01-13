Data Fetching:
--> Kraken API retrieves 30 days of hourly OHLC data
--> Features: close price, SMA_20 (Simple Moving Average), RSI (Relative Strength Index)
--> Error handling for API timeouts and data validation

Data Preparation:
--> StandardScaler normalizes features independently
--> Creates 48-hour time series sequences
--> 80% training, 20% testing split
--> Handles NaN and infinity values with clipping (-10, 10)

Model Architecture:
--> CNN: Two layers (32, 64 filters) with BatchNorm and LeakyReLU
--> LSTM: 2-layer bidirectional (64 input, 32 hidden units)
--> Dropout (0.1) and BatchNorm for regularization
--> Final dense layers: 64 -> 32 -> 1

Training Process:
--> AdamW optimizer (lr=0.001, weight_decay=0.01)
--> ReduceLROnPlateau scheduler with factor 0.5
--> Gradient clipping at 1.0
--> Early stopping with patience=5
--> Batch size=32, max epochs=20

Performance Monitoring:
--> Training and validation loss tracking
--> MSE loss function
--> R² score evaluation
--> Mean Absolute Error calculation
--> Visualization of predictions vs actual prices

Additional Features:
--> Baseline comparison with last known price
--> Real-time price visualization
--> Training history plots
--> Error metrics in USD format
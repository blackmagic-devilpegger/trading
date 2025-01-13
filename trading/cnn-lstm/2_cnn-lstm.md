Data Fetching:
--> Kraken API retrieves 30 days of hourly OHLC data
--> Features: close price, SMA_20 (Simple Moving Average), RSI (Relative Strength Index)
--> Proper error handling for API timeouts and connection issues
--> Automatic data validation and cleaning

Data Preparation:
--> Separate scalers for price (target) and technical indicators
--> Creates 48-hour sequences for time series prediction
--> Train-test split: 80% training, 20% testing
--> Robust handling of NaN and infinity values

Model Architecture:
--> CNN layers: two convolutional layers (32, 64 filters)
--> LSTM layers: bidirectional with 2 layers, 32 hidden units
--> Batch normalization for training stability
--> Dropout (0.1) for regularization
--> LeakyReLU activation functions

Training Process:
--> AdamW optimizer with weight decay (0.01)
--> Learning rate scheduling with ReduceLROnPlateau
--> Gradient clipping at 1.0
--> Early stopping with patience of 5
--> Batch size of 32 and maximum 20 epochs

Price Prediction:
--> Independent scaling of price data
--> Proper inverse transformation for predictions
--> Last value baseline comparison
--> Direct price prediction in USD

Performance Metrics:
--> Mean Squared Error (MSE)
--> Root Mean Squared Error (RMSE)
--> Mean Absolute Error (MAE)
--> R² Score for model fit
--> Error analysis with visualization

Visualization:
--> Training and validation loss curves
--> Actual vs predicted prices in USD
--> Prediction error analysis
--> Price-formatted axis labels
--> Separate error distribution plot

Output Features:
--> Real-time price predictions
--> Comparison with last known price
--> Detailed error metrics
--> Performance statistics in USD
--> Sample prediction analysis
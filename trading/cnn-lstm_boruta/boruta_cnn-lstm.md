Short description: 
This model utilizes Boruta-selected features combined with a hybrid CNN-LSTM architecture 
to predict Bitcoin prices, aiming for improved feature relevance and prediction accuracy. 
The use of feature selection and RandomForest-based rankings ensures the inclusion of only 
the most significant inputs for the model.

Data Fetching:
--> Kraken API retrieves 30 days of hourly OHLC data
--> Features: open, high, low, close, volume, SMA (20), RSI, MA (50), volatility
--> Proper error handling for API timeouts and connection issues
--> Automatic data validation and cleaning

Data Preparation:
--> Boruta feature selection for optimal feature set
--> RandomForest-based feature importance ranking
--> Separate scalers for price and selected features
--> Creates 48-hour sequences for time series prediction
--> Train-test split: 80% training, 20% testing

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
--> Feature-based prediction using Boruta-selected inputs
--> Proper inverse transformation for predictions
--> Last value baseline comparison
--> Direct price prediction in USD

Performance Metrics:
--> Mean Squared Error (MSE)
--> R² Score for model fit
--> Mean Absolute Error (MAE)

Visualization:
--> Training and validation loss curves
--> Actual vs predicted prices plot
--> Last value prediction comparison
--> Grid overlay for better readability

Key Differences from Your Template:
1. Features: Your code includes Boruta feature selection
2. Metrics: Doesn't include RMSE or error distribution plots
3. Visualization: Simpler plotting without price-formatted axis labels
4. Output: No real-time predictions, focuses on historical analysis


Output: 
[...]
MSE: $9033470.00
R² Score: -0.0711
Mean Absolute Error: $2392.57
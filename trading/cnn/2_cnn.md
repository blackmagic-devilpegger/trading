Data Fetching:
--> Kraken API retrieves 30 days of hourly OHLC data
--> Features: close, SMA_20, RSI
--> Error handling added for API timeouts and data validation

Data Preparation:
--> StandardScaler normalizes features
--> Creates 48-hour sequences for time series data
--> Train-test split: 80% training, 20% testing
--> Handles NaN and infinity values for data robustness

Model Architecture (BitcoinCNN):
--> Input: 3 channels (close, SMA_20, RSI)
--> Convolutional Layers:
    --> 3 Conv1d layers with kernel sizes 3, 5, and 7
    --> BatchNorm, ReLU activation, and Dropout (0.2) after each layer
--> Attention Mechanism:
    --> Attention Conv1d layer (96→64 channels)
    --> Fully connected attention weights with Softmax
--> Dense Layers:
    --> Flattened output (64×48) → 128 → 64 → 1
    --> BatchNorm, ReLU, and Dropout (0.2) after dense layers

Training:
--> Loss Function: Mean Squared Error (MSE)
--> Optimizer: AdamW (learning rate 0.001, weight decay 0.01)
--> Scheduler: OneCycleLR with gradual learning rate adjustment
--> Batch size: 32, Epochs: 20
--> Gradient clipping (max norm = 1.0) and early stopping (patience = 5)

Performance Monitoring:
--> Tracks training, validation, and baseline (last hour) losses
--> Evaluates Mean Absolute Error (MAE), MSE, R² score, and error standard deviation
--> Saves the best model state during training

Visualization:
--> Plots:
    --> Training, validation, and baseline losses
    --> Learning rate schedule
    --> Actual vs. predicted prices
    --> Prediction errors over time

Evaluation:
--> Rescales predictions and actual values
--> Compares model predictions against a last-hour baseline
--> Prints detailed metrics for model and baseline performance

Output:
[...]
Prediction Statistics:
Model MSE: $6891924.80
Baseline MSE: $279204.56
Model R² Score: 0.2480
Baseline R² Score: 0.9695
Model Mean Absolute Error: $2380.33
Baseline Mean Absolute Error: $353.04
Model Standard Deviation of Error: $2621.19
Baseline Standard Deviation of Error: $527.03
Model Improvement over Baseline: -2368.41%

=> Model Performance Comparison (to 1_cnn.py):  
The model's performance significantly deteriorated in the second evaluation, as indicated by higher MSE and MAE values.
--> Model Performance: MSE increased from $2181532.63 to $6891924.80, indicating worse accuracy.  
--> Prediction Consistency: R² dropped from 0.7620 to 0.2480, with error variability rising from $1240.91 to $2621.19.  
--> Baseline Stability: Baseline MSE and MAE remained consistent, showing reliable last-hour predictions.  
--> Performance Gap: Larger gap between model and baseline, with MAE increasing by over $1000.  
=> Conclusion: Significant regression in model performance.
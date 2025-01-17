CNN:
Short description: 
The CNN model utilizes convolutional layers to extract temporal and spatial 
patterns from Bitcoin price data, with a focus on improving prediction accuracy 
through feature normalization, dropout regularization, and detailed performance 
monitoring. Despite its advanced architecture, the model struggles with prediction 
variability and underperforms compared to simpler baselines and other iterations 
like the LSTM model.

Data Fetching:
--> Kraken API retrieves 30 days of hourly OHLC data
--> Features: close, SMA_20, RSI
--> Error handling added for API timeouts and data validation

Data Preparation:
--> StandardScaler for feature normalization
--> 48-hour sequences with proper handling of NaN and infinity values
--> 80/20 train/test split
--> Added data validation checks

Model Architecture
--> Two-layer CNN:
  - Conv1d (3→16, kernel=3) with BatchNorm
  - Conv1d (16→32, kernel=3) with BatchNorm
--> Dropout rate: 0.1
--> Dense layers: 32*sequence_length → 64 → 1
--> Numerical stability improvements (epsilon addition)

Training
--> Loss: MSE
--> Optimizer: Adam (lr=0.0001, weight_decay=1e-5)
--> Batch size: 32
--> Early stopping with patience=5
--> Gradient clipping at 1.0

Performance Monitoring
--> Tracks three losses:
  - Training loss
  - Validation loss
  - Baseline (last hour) loss
--> Saves best model state
--> Comprehensive error analysis:
  - MSE
  - R² score
  - Mean Absolute Error
  - Error standard deviation
--> Baseline comparison with improvement percentage

Visualization
--> Three-panel visualization:
  - Loss curves (training/validation/baseline)
  - Price predictions vs actual
  - Error trends (model vs baseline)
--> Detailed metrics printout

Key Improvements
- Early stopping mechanism
- Better baseline integration
- Robust error handling
- Separated visualization and metrics functions
- Type hints and proper logging
- Memory efficiency improvements

Current Limitations
- Model often underperforms simple last-hour baseline
- High prediction variance
- Sensitive to market volatility

Output:
[...]
Prediction Statistics:
Model MSE: $2181532.63
Baseline MSE: $279159.48
Model R² Score: 0.7620
Baseline R² Score: 0.9695
Model Mean Absolute Error: $1230.60
Baseline Mean Absolute Error: $352.95
Model Standard Deviation of Error: $1240.91
Baseline Standard Deviation of Error: $526.98
Model Improvement over Baseline: -681.46%


=> Model Performance Comparison (to 2_lstm.py):
Accuracy: other model (2_lstm.py) significantly better accuracy compared to this model (based on MSE and MAE)
Error Variability: probably higher standard deviation of error ($1,240.91) compared to the other (likely lower, given the smaller MSE and MAE)
Prediction Stability: other more stable predictive performance, indicated by its relatively lower error metrics across all measures
Conclusion:
The other model outperforms this in all key metrics, highlighting a regression in predictive quality in the latter iteration.
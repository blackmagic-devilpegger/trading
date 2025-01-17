# trading

Trading Project: CNN, LSTM, RNN Experiments

General Description and Goal of the Project
This project explores advanced deep learning models (CNN, LSTM, RNN, and hybrid CNN-LSTM architectures) for predicting Bitcoin prices based on historical data. By leveraging feature engineering, temporal patterns, and optimization strategies, the primary objective is to develop robust trading models that integrate regression and classification tasks to:

- Predict Bitcoin's next closing price (regression).
- Classify price movements into rise, fall, or stability (classification).

The idea for this project is based on insights from the paper "Machine learning-based predictive modeling of Bitcoin prices using blockchain information," which discusses leveraging advanced models for financial forecasting (https://jfin-swufe.springeropen.com/counter/pdf/10.1186/s40854-024-00643-1.pdf). The paper proposed using the Boruta algorithm alongside a CNN-LSTM model, which inspired our decision to integrate this approach into our experiments to enhance feature selection and predictive modeling.

The project uses data fetched from Kraken and focuses on creating efficient, accurate, and interpretable trading strategies.

Summary of Experiments and Results

1. RNN Experiments
- Goal: Develop a foundational understanding of simple recurrent architectures.
- Key Features:
  - Basic RNN layers with standard activation functions.
  - Added Dropout layers to regularize.
- Results:
  - Limited performance due to vanishing gradients and lack of long-term memory.

2. LSTM Experiments
- Goal: Leverage temporal dependencies in sequential data for improved price prediction.
- Key Features:
  - Single and stacked LSTM layers.
  - Integration of dropout to mitigate overfitting.
- Results:
  - Demonstrated better performance compared to CNN in capturing temporal trends.
  - Validation loss showed gradual reduction across epochs, indicating improved generalization.

3. CNN Experiments
- Goal: Use convolutional layers to capture temporal and spatial patterns.
- Key Features:
  - Conv1d layers for feature extraction.
  - Dropout for regularization.
  - Dense layers for regression outputs.
- Results:
  - Achieved moderate prediction accuracy but suffered from high prediction variance.
  - Underperformed compared to simple baselines (e.g., last-hour prediction).
    
4. CNN-LSTM Hybrid
- Goal: Combine CNN's feature extraction with LSTM's sequence modeling for enhanced performance.
- Key Features:
  - Multi-scale convolutional layers with kernel sizes (3, 5, 7).
  - LSTM layers for sequence modeling.
  - Attention mechanism for dynamic feature weighting.
- Results:
  - Outperformed standalone CNN and LSTM models in capturing complex trends.
  - Showed the best generalization but had high computational costs.

5. Feature Engineering with Boruta (RNN-Boruta)
- Goal: Identify the most relevant features using Boruta algorithm for regression and classification.
- Key Features:
  - Engineered features: RSI, SMA 20/50, and volatility.
  - Boruta for feature selection.
  - RNN layers with selected features.
- Results:
  - Reduced model complexity by eliminating irrelevant features.
  - Regression models showed improved validation loss.
  - Classification models failed to identify significant features, highlighting challenges in capturing nonlinear patterns.

Key Findings

1. Feature Engineering: Adding technical indicators (e.g., RSI, SMA) improved model performance but required careful selection to avoid overfitting.
2. Hybrid Models: CNN-LSTM hybrids demonstrated superior performance in trend detection and forecasting compared to standalone models.
3. Baselines: Simple baselines (e.g., last-hour prediction) often outperformed complex models in terms of mean absolute deviation (MAD), indicating challenges in generalization.
4. Attention Mechanism: Integrated attention layers in CNN-LSTM models dynamically weighted features, improving the interpretation of temporal patterns.
5. Metrics Tracking: Comprehensive tracking of MSE, MAE, R², and baseline comparison highlighted model strengths and limitations.

This project lays a foundation for exploring advanced deep learning techniques in financial time series forecasting. Despite challenges in outperforming baselines, it provides valuable insights into combining feature engineering, hybrid architectures, and attention mechanisms for robust trading models.

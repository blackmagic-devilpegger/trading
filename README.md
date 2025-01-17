### Bitcoin Trading Project: Deep Learning Models Analysis

## Project Overview
A comprehensive exploration of deep learning architectures for Bitcoin price prediction, implementing various neural network models including CNN, LSTM, RNN, and hybrid architectures. The project aims to develop robust trading models for both regression (price prediction) and classification (movement direction) tasks.

The idea for this project is based on insights from the paper "Machine learning-based predictive modeling of Bitcoin prices using blockchain information," which discusses leveraging advanced models for financial forecasting (https://jfin-swufe.springeropen.com/counter/pdf/10.1186/s40854-024-00643-1.pdf). The paper proposed using the Boruta algorithm alongside a CNN-LSTM model, which inspired our decision to integrate this approach into our experiments to enhance feature selection and predictive modeling.

The project uses data fetched from Kraken and focuses on creating efficient, accurate, and interpretable trading strategies.

## Summary of Experiments and Results
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

Key Finding: Simple baselines (e.g., last-hour prediction) often outperformed complex models in terms of mean absolute deviation (MAD), indicating challenges in generalization.

## Best Model Comparisons

These models represent the best experiments for their respective architectures, showcasing the strengths and weaknesses of various approaches like RNN, LSTM, CNN, and hybrid models. LSTM_2 stands out as the best-performing LSTM model, and RNN_2 delivers impressive results among RNN-based approaches.

## Model Performance Table

| Model                   | MSE (USD)      | MAE (USD)      | MAD Predictions (USD) | MAD Baseline (USD) | R² Score | Notes              |
|-------------------------|----------------|----------------|------------------------|--------------------|----------|--------------------|
| RNN_2                  | 1,566,547.71   | 939.28         | 6,901.52              | 2,942.34           | -        | Best RNN model     |
| LSTM_2                 | 1,194,710.50   | 829.62         | 829.62                | 284.67             | -        | Best LSTM model    |
| CNN_1                  | 2,181,532.63   | 1,230.60       | 1,240.91              | 526.98             | 0.7620   | Best CNN model     |
| Boruta-RNN Regression  | -              | -              | 5,662.93              | 2,802.36           | -        | Feature selection  |
| Boruta-CNN-LSTM        | 9,033,470.00   | 2,392.57       | -                    | -                  | -0.0711  | Hybrid architecture|

## Key Takeaways
- **Best Overall Model**: LSTM_2 delivers the best performance with the lowest MSE (1,194,710.50 USD) and MAE (829.62 USD), showcasing its effectiveness in capturing temporal dependencies.
- **RNN_2**: Strong performance among RNN models, with an MSE of 1,566,547.71 USD and reasonable MAE (939.28 USD), indicating its capability for sequential data analysis.
- **CNN_1**: Achieves good prediction accuracy with an R² score of 0.7620 but struggles with higher MAD and MSE compared to LSTM_2.
- **Boruta-RNN Regression**: Incorporates feature selection for better input relevance, achieving a MAD of 5,662.93 USD, though improvement over baseline remains limited.
- **Boruta-CNN-LSTM**: High computational complexity with poor performance (MSE: 9,033,470.00 USD, MAE: 2,392.57 USD), indicating overfitting or suboptimal architecture tuning.

## Summary
This comparison highlights the strengths of LSTM-based models for Bitcoin price prediction due to their superior handling of temporal dependencies and lower error metrics. While CNN and hybrid models like Boruta-CNN-LSTM introduce innovative architectures, they often require extensive tuning and may suffer from overfitting or high computational costs. Feature selection using Boruta enhances input relevance but needs careful integration with the chosen architecture for optimal results. Despite challenges in outperforming baselines, it provides valuable insights into combining feature engineering, hybrid architectures, and attention mechanisms for robust trading models.

## Personal opimion
The experiments demonstrate the challenges in achieving robust predictive models for Bitcoin prices. Despite leveraging advanced architectures and feature selection techniques inspired by the referenced paper, the performance consistently fell short of the baseline. While the Boruta-CNN-LSTM model introduced promising innovations, its results were not as strong as anticipated. With additional time and further optimization, there may still be potential to improve its effectiveness and close the gap with the baseline.



Forward Pass Enhancement
more precise, time-step-specific prediction strategy by:

- Generating a separate prediction for each individual input sequence
- Preserving the unique temporal context of each sequence
- Providing a more nuanced assessment of the model's predictive capabilities

set data range 

Baseline Loss: Compares the prediction accuracy of the model against a naive baseline, where the last value in the input sequence is used as the predicted price.

During each epoch, batches of training data are passed through the LSTM model to compute predictions.

Predictions and actual values are rescaled using the scaler’s inverse_transform function.

Prediction deviations are computed by subtracting actual prices from predicted prices.

Baseline MSE:

Compares the model’s predictions against a naive baseline, where the last input value is used as the prediction.
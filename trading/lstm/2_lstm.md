Changes to 1_lstm.py:
Forward Pass Enhancement:  
--> Generates predictions per input sequence, preserving temporal context.  
--> Provides detailed evaluation of model accuracy.  

Data Range:
--> Sets clear training and testing ranges for robust evaluation.  

Baseline Loss:
--> Compares predictions against the last input value as a naive baseline.  

Training Workflow: 
--> Processes training batches, rescales values, and computes deviations between predicted and actual prices.  

Baseline MSE:  
--> Measures prediction accuracy relative to the naive baseline. 

Output:
Data retrieved: 720 rows
Date range: 2024-12-12 21:00:00 to 2025-01-11 20:00:00
Training Start: 2024-12-12 21:00:00, Training End: 2024-12-21 20:00:00
Testing Start: 2024-12-21 21:00:00, Testing End: 2025-01-11 20:00:00
Training Data Points: 216
Testing Data Points: 504
Training data range: 2024-12-12 21:00:00 to 2024-12-21 20:00:00
Training data points: 216
Testing data range: 2024-12-21 21:00:00 to 2025-01-11 20:00:00
Testing data points: 504
[...]
Final Model Evaluation:
Model MSE: 1194710.50
Baseline MSE: 186858.23
Model improvement over baseline: -539.37%
Model MAE: 829.62 USD
Baseline MAE: 285.29 USD
Mean Squared Error (MSE): 1194710.5000
Mean Absolute Deviation (MAD) of Predictions: 829.6174 USD
Mean Absolute Deviation (MAD) of Last Hour Price: 284.6669 USD


=> Model Performance Improvement (to 1_lstm.py): The current model shows better accuracy (lower MSE 
and MAD) compared to the previous iteration but still underperforms relative to the baseline.
Baseline Consistency: Baseline predictions remain significantly more reliable, emphasizing the 
model's need for optimization.
Prediction Gap: The gap between the model's performance and the baseline highlights challenges 
in achieving meaningful predictive gains.
Conclusion: While the model's predictive capability has improved, it remains outperformed by 
the simpler last-hour baseline.
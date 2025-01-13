LSTM:
Sequenz length: 48 (48 hours)
--> higher number of sequences: flatter prediction price line
--> lower number of sequences: prediction price line follows the actual data more closely
=> 48 hours: good trade-off between following the actual data and being flat (choice based on MAD)

Hidden Size: 64
--> higher hidden size: more accurate prediction price
--> lower hidden size: lower prediction price

Batch size: 64
--> higher batch size: lower flat prediction price
--> lower batch size: lower prediction price
=> 64: higher and more accurate prediction price

Number of epochs: 20
--> higher number of epochs: more accurate prediction 
--> lower number of epochs: less accurate prediction
=> 20: good trade-off between accuracy and time

Optimizer: Adam
--> SGD: straight lower line
---> RMSprop: flatter prediction line
---> AdamW: higher MAD (than Adam)

Learning Rate: 0.0011
--> higher learning rate: higher MAD, straight line
--> lower learning rate: higher MAD, straight low line
=> lowest MAD with learning rate 0.0011


Output: 
[...]
Average Bitcoin price in the selected period: 94188.50 USD
Mean Squared Error (MSE): 2992430.5000
Mean Absolute Deviation (MAD) of Predictions: 1573.8402 USD
Mean Absolute Deviation (MAD) of Last Hour Price: 350.3919 USD

=> MAD of predictions is 4.5 times higher than MAD of last hour price
The model's MAD is significantly higher than the baseline, indicating room 
for improvement in prediction accuracy compared to the naive last-hour baseline.






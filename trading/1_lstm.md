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

Number of layer? Scaler? More forward passes?
# FTJ-based-memristor-neural-network
This is the code repository for our reserch "A physics-based predictive model for pulse design to realize high-performance memristive neural networks". 
## Introduction
This repository contains PyTorch implementation of neural network train in situ, and the network is train based on FTJ. The Ferroelectric tunnel junction(FTJ)-based neural network uses the LTP/LTD characteristics of the FTJ for the weight update. 
## Dataset
All experiments are done on MNIST, as it provides a training set of 60,000 handwritten digits and a validation set of 10,000 handwritten digits. The images have size 28*28 pixels.
## Usage
To train the model, first we need to propose a model for the pulse design. In this code repository, we choose FTJ. Then, please run<br>

    training_mnist_FTJ.py

The LTP/LTD characteristics of FTJ is import:
```Python
from curve2 import Y
```
For three pulse schemes, Y can be import from `curve1 (scheme1)`, `curve2 (scheme2)` or `curve3 (scheme3)`.
In my experiments, I found that learning rate has a significant impact on the final performance and 0.2 is the learning rate I used (may not be the best).


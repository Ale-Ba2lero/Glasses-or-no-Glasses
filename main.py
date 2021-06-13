import sys
import numpy as np
import matplotlib.pyplot as plt

import nnfs
from nnfs.datasets import spiral_data

from scratch.loss import CategoricalCrossentropy
from scratch.layers import Dense
from scratch.activations import ReLU, Softmax

nnfs.init()

# ------------------------------------ DATASET
X, y = spiral_data(samples=4, classes=3)

# ------------------------------------ NEW NET
dense1 = Dense(2, 3, ReLU())
dense2 = Dense(3, 3, Softmax())

dense1.forward(X)
dense2.forward(dense1.output)

loss_function = CategoricalCrossentropy()
loss = loss_function.calculate(dense2.output, y)

print("Loss: ", loss)

# 1) compute the gradient on scores

# 2) backpropate the gradient to the parameters for each layer and activation function

#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.show()
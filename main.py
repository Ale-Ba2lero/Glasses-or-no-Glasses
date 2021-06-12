import sys
import numpy as np
import matplotlib as plt

import nnfs
from nnfs.datasets import spiral_data

from scratch.loss import CategoricalCrossentropy
from scratch.layers import Dense
from scratch.activations import ReLU, Softmax

nnfs.init()

X, y = spiral_data(samples=10, classes=3)

# ------------------------------------ NEW NET
dense1 = Dense(2, 3, ReLU())
dense2 = Dense(3, 3, Softmax())

dense1.forward(X)
dense2.forward(dense1.output)

loss_function = CategoricalCrossentropy()
loss_new = loss_function.calculate(dense2.output, y)

print("Loss: ", loss_new)

'''
# ------------------------------------ OLD NET
dense1_old = Dense(2, 3)
activation1 = ReLU()
dense2_old = Dense(3, 3)
activation2 = Softmax()

dense1_old.forward(X)
activation1.forward(dense1.output)
dense2_old.forward(activation1.output)
activation2.forward(dense2.output)

loss_function = CategoricalCrossentropy()
loss_old = loss_function.calculate(activation2.output, y)

print("Old: \n", loss_old)

'''
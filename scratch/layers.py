

import numpy as np

class Dense:
    def __init__(self, inputs, neurons):
        self.W = 0.10 * np.random.randn(inputs, neurons)
        self.b = np.zeros((1, neurons))

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.W) + self.b

    def backward(self, dscore):
        self.dW = np.dot(self.inputs.T, dscore)
        self.db = np.sum(dscore, axis=0, keepdims=True)

    def update(self, step_size):
        self.W += -step_size * self.dW
        self.b += -step_size * self.db
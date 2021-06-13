

import numpy as np

class Dense:
    def __init__(self, inputs, neurons, activation=False):
        self.weights = 0.10 * np.random.randn(inputs, neurons)
        self.biases = np.zeros((1, neurons))
        if activation is not False:
            self.activation = activation

    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases
        if hasattr (self, "activation"):
            self.output = self.activation.compute(self.output)

    def backward(self, dscore):
        self.dweights = np.dot(self.weights.T, dscore)
        self.dbiases = np.sum(dscore, axis=0, keepdims=True)

        # TODO propagate also through the activation function

    def update(self, step_size):
        self.weights += -step_size * self.dweights
        self.biases += -step_size * self.dbiases
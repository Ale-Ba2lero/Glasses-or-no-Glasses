
from scratch.activations import Activation
import numpy as np

from scratch.activations import ReLU

class Dense:
    def __init__(self, n_inputs, n_neurons, activation=False):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        if activation is not False:
            self.activation = activation

    def forward(self, inputs):
        self.dot = np.dot(inputs, self.weights) 
        self.output = self.dot + self.biases

        if hasattr (self, "activation"):
            self.output = self.activation.compute(self.output)

    def backward(self, df):
        self.dweights = self.weights * df
        self.dbiasses = self.biases * df

import numpy as np
from numpy.lib.function_base import select

class Activation:
    def compute(self, inputs):
        self.forward(inputs)
        return self.output

class ReLU(Activation):
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Softmax(Activation):
    def forward(self, inputs):
        # subtract the max (prevent overflow) and exponentialize
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalization
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities 
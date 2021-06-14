
import numpy as np

class Activation:
    def compute(self, inputs):
        self.forward(inputs)
        return self.output

    def backpropagation(self, dscore, layer):
        self.backward(dscore, layer)
        return self.dlayer

class ReLU(Activation):
    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.maximum(0, inputs)

    def backward(self, dscore, layer):
        self.dlayer = np.dot(dscore, layer.T)
        self.dlayer[self.output <= 0] = 0

class Softmax(Activation):
    def forward(self, inputs):
        self.num_examples = len(inputs)
        # subtract the max (prevent overflow) and exponentialize
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        # normalization
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities 

    def backward(self):
        # TODO mettere qui calcolo derivata loss
        pass

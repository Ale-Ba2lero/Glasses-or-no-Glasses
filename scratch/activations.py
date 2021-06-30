import numpy as np


class ReLU:
    def compute(self, inputs):
        self.output = np.maximum(0, inputs)
        #    print (f'ReLU ({self.inputs.shape}) = {self.output.shape}')
        return self.output

    def backpropagation(self, dscore, layer):
        dlayer = np.dot(dscore, layer.T)
        dlayer[self.output <= 0] = 0
        #    print (f'ReLU (dscore{dscore.shape} * layer.T{layer.T.shape}) = {self.dlayer.shape}')
        return dlayer


class Softmax:
    def compute(self, inputs):
        # subtract the max (prevent overflow) and exponentialize
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))

        # normalization
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)

        return probabilities

    def backpropagation(self):
        # the CategoricalCrossentropy Loss function already takes into account
        # the softmax backpropagation
        pass

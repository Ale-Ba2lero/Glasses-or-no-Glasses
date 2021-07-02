import numpy as np


class Activation:

    def compute(self, inputs: np.ndarray) -> np.ndarray:
        pass

    def backpropagation(self, d_score: np.ndarray, layer: np.ndarray) -> np.ndarray:
        pass


class ReLU(Activation):
    def __init__(self):
        self.output = None

    def compute(self, inputs: np.ndarray) -> np.ndarray:
        self.output: np.ndarray = np.maximum(0, inputs)
        return self.output

    def backpropagation(self, d_score: np.ndarray, layer: np.ndarray) -> np.ndarray:
        d_layer = np.dot(d_score, layer.T)
        d_layer[self.output <= 0] = 0
        return d_layer


class Softmax(Activation):
    def compute(self, inputs: np.ndarray) -> np.ndarray:
        # subtract the max (prevent overflow)
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        return probabilities

    def backpropagation(self, d_score=None, layer=None) -> None:
        # in this implementation the Categorical Cross Entropy Loss function
        # already takes into account the softmax backpropagation
        pass

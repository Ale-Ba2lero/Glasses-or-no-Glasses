import numpy as np

np.seterr(all='raise')


class Activation:

    def compute(self, inputs: np.ndarray) -> np.ndarray:
        pass

    def backpropagation(self, d_score: np.ndarray) -> np.ndarray:
        pass


class ReLU(Activation):
    def __init__(self):
        self.output = None

    def compute(self, inputs: np.ndarray) -> np.ndarray:
        self.output: np.ndarray = np.maximum(0, inputs)
        return self.output

    def backpropagation(self, d_score: np.ndarray) -> np.ndarray:
        d_score[self.output <= 0] = 0
        return d_score


class LeakyReLU(Activation):
    def __init__(self, param=1e-2):
        self.param = param
        self.output = None

    def compute(self, inputs: np.ndarray) -> np.ndarray:
        self.output: np.ndarray = np.where(inputs > 0, inputs, inputs * self.param)
        return self.output

    def backpropagation(self, d_score: np.ndarray) -> np.ndarray:
        d_score[self.output <= 0] = self.param * d_score[self.output]
        return d_score


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

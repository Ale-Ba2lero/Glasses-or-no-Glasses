import numpy

from model.layers.layer import Layer, LayerType
import numpy as np


class LeakyReLU(Layer):

    def __init__(self, alpha=1e-2):
        super().__init__()
        self.alpha = alpha
        self.layer_type = LayerType.RELU
        self.output = None

    def setup(self, input_shape: tuple) -> None:
        self.input_shape = input_shape
        self.output_shape = input_shape

    def forward(self, input_layer: numpy.ndarray) -> numpy.ndarray:
        self.output: np.ndarray = np.where(input_layer > 0, input_layer, input_layer * self.alpha)
        return self.output

    def backpropagation(self, d_score: np.ndarray) -> np.ndarray:
        d_score[self.output <= 0] *= self.alpha
        return d_score

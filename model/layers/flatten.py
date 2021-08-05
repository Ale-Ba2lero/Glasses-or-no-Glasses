from model.layers.layer import Layer, LayerType
import numpy as np


class Flatten(Layer):

    def __init__(self):
        super().__init__()
        self.layer_type = LayerType.FLATTEN

    def setup(self, input_shape: (int, int, int)) -> None:
        self.input_shape: tuple = input_shape
        h, w, d = input_shape
        self.output_shape: int = h * w * d

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.batch_size = inputs.shape[0]
        output = inputs.reshape((self.batch_size,) + (self.output_shape,))
        return output

    def backpropagation(self, d_score: np.ndarray) -> np.ndarray:
        d_score = d_score.reshape((self.batch_size,) + self.input_shape)
        return d_score

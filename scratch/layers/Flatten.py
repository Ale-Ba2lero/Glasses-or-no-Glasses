from scratch.layers.Layer import Layer, LayerType
import numpy as np


class Flatten(Layer):

    def __init__(self):
        super().__init__()
        self.layer_type = LayerType.FLATTEN
        self.next_layer = None

    def setup(self, input_shape: tuple[int, int, int, int], next_layer: np.ndarray = None) -> None:
        self.input_shape: tuple = input_shape
        self.next_layer: np.ndarray = next_layer
        batch, h, w, d = input_shape
        self.output_shape: tuple[int, int] = (batch, h * w * d)
        print(f"Flatten layer\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\n")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        # Hope reshape works as I expect
        output = inputs.reshape(self.output_shape)
        return output

    def backpropagation(self, d_score: np.ndarray) -> np.ndarray:
        d_score = np.dot(d_score, self.next_layer.W.T)
        d_score = d_score.reshape(self.input_shape)
        return d_score

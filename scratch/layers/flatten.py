from scratch.layers.layer import Layer, LayerType
import numpy as np


class Flatten(Layer):

    def __init__(self):
        super().__init__()
        self.layer_type = LayerType.FLATTEN
        self.next_layer = None

    def setup(self, input_shape: (int, int, int), next_layer: Layer = None) -> None:
        self.input_shape: tuple = input_shape
        self.next_layer: Layer = next_layer
        h, w, d = input_shape
        self.output_shape: (int,) = (h * w * d, )
        # print(f"Flatten layer\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\n")

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.batch_size = inputs.shape[0]
        output = inputs.reshape((self.batch_size,) + self.output_shape)
        return output

    def backpropagation(self, d_score: np.ndarray) -> np.ndarray:
        d_score = np.dot(d_score, self.next_layer.W.T)
        d_score = d_score.reshape((self.batch_size,) +self.input_shape)
        return d_score

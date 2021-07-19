import numpy as np

from model.layers.layer import Layer, LayerType


class Dropout(Layer):
    def __init__(self, prob: float = 0.2):
        super().__init__()
        self.layer_type: LayerType = LayerType.DROPOUT
        self.prob = prob

    def setup(self, input_shape: tuple) -> None:
        self.output_shape = input_shape

    def forward(self, input_layer: np.ndarray, train: bool = True) -> np.ndarray:
        if train:
            flat = np.array(input_layer).flatten()
            random_indices = np.random.randint(0, len(flat), int(self.prob * len(flat)))
            flat[random_indices] = 0
            output = flat.reshape(input_layer.shape)
        else:
            output = input_layer

        return output

    def backpropagation(self, d_score: np.ndarray) -> np.ndarray:
        return d_score

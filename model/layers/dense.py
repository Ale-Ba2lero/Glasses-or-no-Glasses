from model.activations import Activation, ReLU
from model.layers.layer import Layer, LayerType
import numpy as np


class Dense(Layer):
    def __init__(self, num_neurons, activation=None) -> None:
        super().__init__()
        self.num_neurons: int = num_neurons
        self.activation: Activation = activation
        self.layer_type: LayerType = LayerType.DENSE

        self.batch_size = None
        self.W = None
        self.dW = None
        self.b = None
        self.db = None
        self.next_layer = None
        self.input_layer = None

    def setup(self, input_shape: int, next_layer: Layer = None) -> None:
        # / np.sqrt(self.input_shape) <- Xavier initialization
        self.W: np.ndarray = np.random.randn(input_shape, self.num_neurons) / np.sqrt(input_shape)
        self.b: np.ndarray = np.zeros((1, self.num_neurons))
        self.next_layer: Layer = next_layer
        self.output_shape: int = self.num_neurons

    def forward(self, input_layer) -> np.ndarray:
        self.batch_size = input_layer.shape[0]
        self.input_layer: np.ndarray = input_layer
        output = np.dot(input_layer, self.W) + self.b
        return output

    def backpropagation(self, d_score) -> np.ndarray:
        self.dW: np.ndarray = np.dot(self.input_layer.T, d_score)
        self.db: np.ndarray = np.sum(d_score, axis=0, keepdims=True)
        d_score = np.dot(d_score, self.W.T)
        return d_score

    def update(self, step_size: float = 1e-0) -> None:
        self.W += -step_size * self.dW
        self.b += -step_size * self.db

from scratch.activations import Activation, ReLU
from scratch.layers.Layer import Layer, LayerType
import numpy as np


class Dense(Layer):
    def __init__(self, num_neurons, activation=ReLU()) -> None:
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

    def setup(self, input_shape, next_layer=None) -> None:
        self.input_shape = input_shape
        self.batch_size: int = input_shape[1]
        # multiply by 0.1 to reduce the variance of our initial values
        self.W: np.ndarray = 0.1 * np.random.randn(self.batch_size, self.num_neurons)
        self.b: np.ndarray = np.zeros((1, self.num_neurons))
        self.next_layer: np.ndarray = next_layer
        self.output_shape: tuple[int, int] = (self.batch_size, self.num_neurons)

        print(
            f"Dense layer {self.num_neurons} neurons\ninput size: {input_shape}\nLayer shape: {self.W.shape}\nOutput "
            f"size: {self.output_shape}\n")

    def forward(self, input_layer) -> np.ndarray:
        self.input_layer: np.ndarray = input_layer
        output = np.dot(input_layer, self.W) + self.b
        output = self.activation.compute(output)
        return output

    def backpropagation(self, d_score) -> np.ndarray:
        if self.next_layer is not None:
            d_score = self.activation.backpropagation(d_score, self.next_layer.W)
        self.dW: np.ndarray = np.dot(self.input_layer.T, d_score)
        self.db: np.ndarray = np.sum(d_score, axis=0, keepdims=True)
        self.update()
        return d_score

    def update(self, step_size: float = 1e-0) -> None:
        self.W += -step_size * self.dW
        self.b += -step_size * self.db

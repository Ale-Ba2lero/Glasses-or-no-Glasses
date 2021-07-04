from scratch.activations import Activation, ReLU
from scratch.layers.layer import Layer, LayerType
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

    def setup(self, input_shape: int, next_layer: Layer = None) -> None:
        self.input_shape = input_shape
        # multiply by 0.1 to reduce the variance of our initial values
        self.W: np.ndarray = 0.1 * np.random.randn(self.input_shape, self.num_neurons)
        self.b: np.ndarray = np.zeros((1, self.num_neurons))
        self.next_layer: Layer = next_layer
        self.output_shape: int = self.num_neurons

        # print(
        #    f"Dense layer {self.num_neurons} neurons\ninput size: {input_shape}\nLayer shape: {self.W.shape}\nOutput "
        #    f"size: {self.output_shape}\n")

    def forward(self, input_layer) -> np.ndarray:
        self.input_layer: np.ndarray = input_layer
        output = np.dot(input_layer, self.W) + self.b
        output = self.activation.compute(output)
        return output

    def backpropagation(self, d_score) -> np.ndarray:
        if self.next_layer is not None:
            d_score = self.activation.backpropagation(d_score)
        self.dW: np.ndarray = np.dot(self.input_layer.T, d_score)
        # TODO check db computation
        self.db: np.ndarray = np.sum(d_score, axis=0, keepdims=True)
        d_score = np.dot(d_score, self.W.T)
        return d_score

    def update(self, step_size: float = 1e-0) -> None:
        self.W += -step_size * self.dW
        self.b += -step_size * self.db

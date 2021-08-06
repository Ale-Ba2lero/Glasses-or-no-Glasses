from model.layers.layer import Layer, LayerType
from model.utility import xavier_initialization
import numpy as np


class Dense(Layer):
    def __init__(self, num_neurons) -> None:
        super().__init__()
        self.num_neurons: int = num_neurons
        self.layer_type: LayerType = LayerType.DENSE

        self.batch_size = None
        self.W = None
        self.dW = None
        self.b = None
        self.db = None
        self.next_layer = None
        self.input_layer = None

    def setup(self, input_shape: int, next_layer: Layer = None) -> None:
        self.W: np.ndarray = xavier_initialization(units=self.num_neurons, shape=(input_shape, self.num_neurons))
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

    def update(self, step_size: float = 1e-0, clip=None) -> None:
        if clip is not None:
            self.dW = np.clip(self.dW, -clip, clip)
            self.db = np.clip(self.db, -clip, clip)

        self.W += -step_size * self.dW
        self.b += -step_size * self.db

    def get_deltas(self):
        return self.dW, self.db

    def set_deltas(self, dW, db):
        self.dW = dW
        self.db = db

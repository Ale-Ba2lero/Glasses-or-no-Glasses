from enum import Enum, auto
import abc
import numpy


class LayerType(Enum):
    DENSE = auto()
    CONV = auto()
    MAXPOOL = auto()
    FLATTEN = auto()
    RELU = auto()
    SOFTMAX = auto()
    DROPOUT = auto()


class Layer(abc.ABC):

    def __init__(self):
        self.layer_type = None
        self.input_shape = None
        self.output_shape = None
        self.batch_size = None

    @abc.abstractmethod
    def setup(self, input_shape: tuple) -> None:
        pass

    @abc.abstractmethod
    def forward(self, input_layer: numpy.ndarray) -> numpy.ndarray:
        pass

    @abc.abstractmethod
    def backpropagation(self, d_score: numpy.ndarray) -> numpy.ndarray:
        pass

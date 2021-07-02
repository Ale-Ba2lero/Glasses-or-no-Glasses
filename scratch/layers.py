from scratch.activations import Activation, ReLU
import numpy as np
from enum import Enum, auto
import abc

class LayerType(Enum):
    DENSE = auto()
    CONV = auto()
    MAXPOOL = auto()
    FLATTEN = auto()

class Layer(abc.ABC):

    @abc.abstractmethod
    def setup(self, input_shape: tuple, y_true: np.array) -> (np.array, float):
        pass

class Dense:
    def __init__(self, num_neurons, activation=ReLU()) -> None:
        self.num_neurons: int = num_neurons
        self.activation: Activation = activation
        self.layer_type: LayerType = LayerType.DENSE

        self.batch_size = None
        self.W = None
        self.dW = None
        self.b = None
        self.db = None
        self.next_layer = None
        self.output_shape = None
        self.input_layer = None

    def setup(self, input_shape, next_layer=None) -> None:
        self.batch_size: int = input_shape[1]
        # multiply by 0.1 to reduce the variance of our initial values
        self.W: np.array = 0.1 * np.random.randn(self.batch_size, self.num_neurons)
        self.b: np.array = np.zeros((1, self.num_neurons))
        self.next_layer: np.array = next_layer
        self.output_shape: tuple = (self.batch_size, self.num_neurons)

        print(
            f"Dense layer {self.num_neurons} neurons\ninput size: {input_shape}\nLayer shape: {self.W.shape}\nOutput "
            f"size: {self.output_shape}\n")

    def forward(self, input_layer) -> np.array:
        self.input_layer: np.array = input_layer
        output = np.dot(input_layer, self.W) + self.b
        output = self.activation.compute(output)
        return output

    def backpropagation(self, d_score) -> np.array:
        if self.next_layer is not None:
            d_score = self.activation.backpropagation(d_score, self.next_layer.W)
        self.dW: np.array = np.dot(self.input_layer.T, d_score)
        self.db: np.array = np.sum(d_score, axis=0, keepdims=True)
        self.update()
        return d_score

    def update(self, step_size=1e-0) -> None:
        self.W += -step_size * self.dW
        self.b += -step_size * self.db


class Conv:

    def __init__(self, num_filters, kernel_size=(3, 3), padding=0, stride=1, activation=ReLU(), log=False):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.activation = activation
        self.log = log
        # TODO add bias?
        self.layer_type = LayerType.CONV

    def setup(self, input_shape):
        self.input_shape = input_shape
        self.output_shape = self.convolution_compatibility(input_shape)

        # We divide by 10 to reduce the variance of our initial values
        self.filters = np.random.random_sample(
            (self.kernel_size[0], self.kernel_size[1], input_shape[3], self.num_filters)) * 0.1

        print(
            f"Conv layer\ninput: {self.input_shape}\nfilter: {self.filters.shape}\noutput: {self.output_shape}\n")

    def convolution_compatibility(self, input_shape):
        batch, h, w, _ = input_shape
        f_h, f_w = self.kernel_size
        s = self.stride
        p = self.padding

        output_layer_h = (h - f_h + 2 * p) / s + 1
        output_layer_w = (w - f_w + 2 * p) / s + 1

        if output_layer_h % 1 != 0 or output_layer_w % 1 != 0:
            print('Error!: hyperparameters setting is invalid!')
            return None

        return batch, int(output_layer_h), int(output_layer_w), self.num_filters

    def zero_padding(self, inputs, padding=1):
        batch, h, w, d = inputs.shape
        canvas = np.zeros((batch, h + padding * 2, w + padding * 2, d))
        canvas[:, padding:h + padding, padding:w + padding] = inputs
        return canvas

    def iterate_regions(self, inputs):
        """
        Generates all possible  image regions using valid padding.
        - image is a 3d numpy array
        """
        _, h, w, _ = inputs.shape
        h_limit = h - self.kernel_size[0] + 1
        w_limit = w - self.kernel_size[1] + 1

        for i in range(0, h_limit, self.stride):
            for j in range(0, w_limit, self.stride):
                img_region = inputs[:, i:(i + self.kernel_size[0]), j:(j + self.kernel_size[1]), :]
                yield img_region, i, j

    def forward(self, inputs):

        if self.padding > 0:
            inputs = self.zero_padding(inputs, self.padding)
        self.last_input = inputs

        output = np.zeros(self.output_shape)
        b, _, _, d = output.shape

        for idx_b in range(b):
            for idx_d in range(d):
                for img_region, i, j in self.iterate_regions(inputs):
                    output[idx_b, i, j, idx_d] = np.sum(img_region[idx_b] * self.filters[:, :, :, idx_d])

        return output

    def backprop(self, dscore):
        dfilters = np.zeros(self.filters.shape)

        for img_region, i, j in self.iterate_regions(self.last_input):
            for b in range(dscore.shape[0]):
                for f in range(self.num_filters):
                    dfilters[:, :, :, f] += dscore[b, i, j, f] * img_region[b]

        self.update(dscore=dfilters)
        return dscore

    def update(self, dscore, learn_rate=1e-0):
        self.filters = self.filters - (dscore * learn_rate)


class MaxPool:
    def __init__(self):
        self.layer_type = LayerType.MAXPOOL

    def setup(self, input_shape):
        self.input_shape = input_shape
        batch, h, w, d = input_shape
        self.output_shape = (batch, h // 2, w // 2, d)
        print(f"Pool layer\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\n")

    def iterate_regions(self, inputs):
        """
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2d numpy array
        """
        batch_size, h, w, d = self.output_shape
        for i in range(h):
            for j in range(w):
                img_region = inputs[:, (i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2), :]
                yield img_region, i, j

    def forward(self, inputs):
        """
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array with dimensions (batchsize, h / 2, w / 2, 3, num_filters).
        - input is a 3d numpy array with dimensions (h, w, 3, num_filters)
        """
        self.last_input = inputs
        batch_size, h, w, d = inputs.shape
        output = np.zeros((batch_size, h // 2, w // 2, d))

        for img_region, i, j in self.iterate_regions(inputs):
            output[:, i, j, :] = np.amax(img_region, axis=(1, 2))

        return output

    def backprop(self, dscore):
        """
        Performs a backward pass of the maxpool layer.
        Returns the loss gradient for this layer's inputs.
        - dL_dout is the loss gradient for this layer's outputs.
        """
        dinput = np.zeros(self.last_input.shape)

        for img_region, i, j in self.iterate_regions(self.last_input):
            b, h, w, d = img_region.shape
            amax = np.amax(img_region, axis=(1, 2))

            for idx_b in range(b):
                for idx_h in range(h):
                    for idx_w in range(w):
                        for idx_d in range(d):
                            # If the pixel was the max value, copy the gradient to it.
                            if img_region[idx_b, idx_h, idx_w, idx_d] == amax[idx_b, idx_d]:
                                dinput[idx_b, i * 2 + idx_h, j * 2 + idx_w, idx_d] = dscore[idx_b, i, j, idx_d]
        return dinput


class Flatten:

    def __init__(self):
        self.layer_type = LayerType.FLATTEN

    def setup(self, input_shape, next_layer):
        self.input_shape = input_shape
        self.next_layer = next_layer
        batch, h, w, d = input_shape
        self.output_shape = (batch, h * w * d)
        print(f"Flatten layer\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\n")

    def forward(self, inputs):
        # Hope reshape works as I expect
        output = inputs.reshape(self.output_shape)
        return output

    def backprop(self, dscore):
        dscore = np.dot(dscore, self.next_layer.W.T)
        dscore = dscore.reshape(self.input_shape)
        return dscore

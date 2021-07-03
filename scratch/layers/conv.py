from scratch.activations import Activation, ReLU
from scratch.layers.layer import Layer, LayerType
import numpy as np
from tqdm import tqdm


class Conv(Layer):

    def __init__(self, num_filters: int, kernel_size: (int, int) = (3, 3), padding: int = 0, stride: int = 1,
                 activation: Activation = ReLU()):
        super().__init__()
        self.num_filters: int = num_filters
        self.kernel_size: tuple = kernel_size
        self.padding: int = padding
        self.stride: int = stride
        self.activation: Activation = activation
        self.layer_type: LayerType = LayerType.CONV

        # TODO add bias?
        self.filters = None
        self.d_filters = None
        self.last_input = None

    def setup(self, input_shape: tuple):
        self.input_shape: tuple = input_shape
        self.output_shape: tuple = self.convolution_compatibility(input_shape)
        channel_RGB = 3
        # We divide by 10 to reduce the variance of our initial values
        self.filters: np.ndarray = np.random.random_sample(
            (self.kernel_size[0], self.kernel_size[1], channel_RGB, self.num_filters)) * 0.1

        # print(
        #    f"Conv layer\ninput: {self.input_shape}\nfilter: {self.filters.shape}\noutput: {self.output_shape}\n")

    def convolution_compatibility(self, input_shape: tuple) -> (int, int, int):
        h, w, _ = input_shape
        f_h, f_w = self.kernel_size
        s = self.stride
        p = self.padding

        output_layer_h = (h - f_h + 2 * p) / s + 1
        output_layer_w = (w - f_w + 2 * p) / s + 1

        if output_layer_h % 1 != 0 or output_layer_w % 1 != 0:
            raise ValueError('Error!: hyper parameters setting is invalid!')

        return int(output_layer_h), int(output_layer_w), self.num_filters

    def zero_padding(self, inputs: np.ndarray, padding: int = 1) -> np.ndarray:
        _, h, w, d = inputs.shape
        canvas = np.zeros((self.batch_size, h + padding * 2, w + padding * 2, d))
        canvas[:, padding:h + padding, padding:w + padding] = inputs
        return canvas

    def iterate_regions(self, inputs: np.ndarray) -> (np.ndarray, int, int):
        _, h, w, num_filters = inputs.shape
        h_limit = h - self.kernel_size[0] + 1
        w_limit = w - self.kernel_size[1] + 1

        for i in range(0, h_limit, self.stride):
            for j in range(0, w_limit, self.stride):
                img_region = inputs[:, i:(i + self.kernel_size[0]), j:(j + self.kernel_size[1]), :]
                yield img_region, i, j

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.batch_size: int = inputs.shape[0]

        if self.padding > 0:
            inputs = self.zero_padding(inputs, self.padding)

        self.last_input: np.ndarray = inputs

        output = np.zeros((self.batch_size,) + self.output_shape)

        volume_depth = output.shape[3]

        for img_region, i, j in self.iterate_regions(inputs):
            for d in range(volume_depth):
                output[:, i, j, d] = np.sum(img_region * self.filters[:, :, :, d], axis=(1, 2, 3))

        return output

    def backpropagation(self, d_score: np.ndarray) -> np.ndarray:
        d_filters = np.zeros(self.filters.shape)

        for img_region, i, j in self.iterate_regions(self.last_input):
            for b in range(d_score.shape[0]):
                for f in range(self.num_filters):
                    d_filters[:, :, :, f] += d_score[b, i, j, f] * img_region[b]

        self.d_filters = d_filters
        return d_score

    def update(self, learn_rate: float = 1e-0) -> None:
        self.filters = self.filters + (-learn_rate * self.d_filters)

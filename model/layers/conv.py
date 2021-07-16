import sys
from model.activations import Activation, ReLU
from model.layers.layer import Layer, LayerType
import numpy as np
from tqdm import tqdm
import time


def iterate_regions(inputs: np.ndarray, kernel: int = 3, stride: int = 1) -> (
        np.ndarray, int, int):
    h: int = 0
    w: int = 0
    img_region = None

    if len(inputs.shape) == 3:
        b, h, w = inputs.shape
        img_region = np.zeros((b, kernel, kernel))
    elif len(inputs.shape) == 4:
        b, h, w, d = inputs.shape
        img_region = np.zeros((b, kernel, kernel, d))

    h_limit = h - kernel + 1
    w_limit = w - kernel + 1

    for i in range(0, h_limit, stride):
        for j in range(0, w_limit, stride):
            if len(inputs.shape) == 3:
                img_region = inputs[:, i:(i + kernel), j:(j + kernel)]
            elif len(inputs.shape) == 4:
                img_region = inputs[:, i:(i + kernel), j:(j + kernel), :]
            yield img_region, i, j


def zero_padding(inputs: np.ndarray, padding: int = 1) -> np.ndarray:
    canvas = None
    if len(inputs.shape) == 3:
        b, h, w = inputs.shape
        canvas = np.zeros((b, h + padding * 2, w + padding * 2))
        canvas[:, padding:h + padding, padding:w + padding] = inputs
    elif len(inputs.shape) == 4:
        b, h, w, d = inputs.shape
        canvas = np.zeros((b, h + padding * 2, w + padding * 2, d))
        canvas[:, padding:h + padding, padding:w + padding] = inputs
    return canvas


class Conv(Layer):

    def __init__(self, num_filters: int, kernel_size: int = 3, padding: int = 0, stride: int = 1,
                 activation: Activation = ReLU()):
        super().__init__()
        self.num_filters: int = num_filters
        self.kernel_size: int = kernel_size
        self.padding: int = padding
        self.stride: int = stride
        self.activation: Activation = activation
        self.layer_type: LayerType = LayerType.CONV

        # TODO add bias?
        self.filters = None
        self.b = None
        self.d_filters = None
        self.last_input = None

        self.f_time = 0
        self.b_time = 0

    def setup(self, input_shape: tuple):
        """if len(input_shape) == 4:
            input_shape = input_shape[1:]"""

        self.input_shape: tuple = input_shape
        self.output_shape: tuple = self.convolution_compatibility(input_shape)

        # / np.sqrt(self.input_shape) <- Xavier initialization
        if len(input_shape) == 2:
            self.filters: np.ndarray = np.random.random_sample(
                (self.kernel_size, self.kernel_size, self.num_filters)) / np.sqrt(self.kernel_size ** 2)

        elif len(input_shape) == 3:
            self.filters: np.ndarray = np.random.random_sample(
                (self.kernel_size, self.kernel_size, self.input_shape[2], self.num_filters)) / np.sqrt(
                self.kernel_size * self.kernel_size * self.input_shape[2])

        """self.b: np.ndarray = np.zeros((1, self.num_filters))"""

    def convolution_compatibility(self, input_shape: tuple) -> (int, int, int):
        h, w = 0, 0
        if len(input_shape) == 2:
            h, w = input_shape
        elif len(input_shape) == 3:
            h, w, _ = input_shape

        s = self.stride
        p = self.padding

        output_layer_h = (h - self.kernel_size + 2 * p) / s + 1
        output_layer_w = (w - self.kernel_size + 2 * p) / s + 1

        if output_layer_h % 1 != 0 or output_layer_w % 1 != 0:
            raise ValueError('Error!: hyper parameters setting is invalid!')

        return int(output_layer_h), int(output_layer_w), self.num_filters

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.last_input: np.ndarray = inputs
        if self.padding > 0:
            inputs = zero_padding(inputs, self.padding)

        batch_size: int = inputs.shape[0]
        output = np.zeros((batch_size,) + self.output_shape)

        start = time.time()
        if len(inputs.shape) == 3:
            for img_region, i, j in iterate_regions(inputs, kernel=self.kernel_size, stride=self.stride):
                for b in range(batch_size):
                    flatten_image = img_region[b].flatten()
                    flatten_filters = self.filters.reshape(self.num_filters, self.kernel_size ** 2)
                    prod_ = (flatten_image * flatten_filters).reshape(self.num_filters, self.kernel_size,
                                                                      self.kernel_size)
                    output[b, i, j, :] = np.sum(prod_, axis=(1, 2))

        end = time.time()
        self.f_time += (end - start)
        return output

    def backpropagation(self, d_score: np.ndarray) -> np.ndarray:
        self.d_filters = np.zeros(self.filters.shape)
        new_d_score = np.zeros(self.last_input.shape)

        # filters delta
        start = time.time()
        batch_size = d_score.shape[0]

        for b in range(batch_size):
            for img_region, i, j in iterate_regions(self.last_input, kernel=self.kernel_size, stride=self.stride):
                for f in range(self.num_filters):
                    if len(self.d_filters.shape) == 3:
                        self.d_filters[:, :, f] += d_score[b, i, j, f] * img_region[b]
                    elif len(self.d_filters.shape) == 4:
                        self.d_filters[:, :, :, f] += d_score[b, i, j, f] * img_region[b]

                # TODO execute this only if there is another layer before
                flat_filters_size = np.prod(self.filters.shape[:-1])
                filter_shape = self.filters.shape[:-1]
                f_filters = self.filters.reshape(flat_filters_size, self.num_filters)
                f_d_score = d_score[b, i, j].flatten()
                prod_ = (f_filters * f_d_score.T).reshape(flat_filters_size, self.num_filters)
                sum_ = np.sum(prod_, axis=1)
                new_d_score[b, i:i + self.kernel_size, j:j + self.kernel_size] = sum_.reshape(filter_shape)

                # new_d_score[b, i:i + self.kernel_size, j:j + self.kernel_size, :] += self.filters[:, :, :, f] * d_score[ b, i, j, f]
        end = time.time()
        self.b_time += (end - start)
        return new_d_score

    def update(self, learn_rate: float = 1e-0) -> None:
        self.filters = self.filters + (-learn_rate * self.d_filters)

    def print_time(self):
        print(f"f-time:{self.f_time} | b-time:{self.b_time}")
        self.f_time = 0
        self.b_time = 0

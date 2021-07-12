import sys

from model.activations import Activation, ReLU
from model.layers.layer import Layer, LayerType
import numpy as np
from tqdm import tqdm
import time


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
        self.b = None
        self.d_filters = None
        self.last_input = None

    def setup(self, input_shape: tuple):
        if len(input_shape) == 4:
            input_shape = input_shape[1:]
        self.input_shape: tuple = input_shape
        self.output_shape: tuple = self.convolution_compatibility(input_shape)

        # / np.sqrt(self.input_shape) <- Xavier initialization
        if len(input_shape) == 2:
            self.filters: np.ndarray = np.random.random_sample(
                (self.kernel_size[0], self.kernel_size[1], self.num_filters)) / np.sqrt(
                self.kernel_size[0] * self.kernel_size[1])
        elif len(input_shape) == 3:
            self.filters: np.ndarray = np.random.random_sample(
                (self.kernel_size[0], self.kernel_size[1], self.input_shape[2], self.num_filters)) / np.sqrt(
                self.kernel_size[0] * self.kernel_size[1] * self.input_shape[2])

        self.b: np.ndarray = np.zeros((1, self.num_filters))

    def convolution_compatibility(self, input_shape: tuple) -> (int, int, int):
        h, w = 0, 0
        if len(input_shape) == 2:
            h, w = input_shape
        elif len(input_shape) == 3:
            h, w, _ = input_shape

        f_h, f_w = self.kernel_size
        s = self.stride
        p = self.padding

        output_layer_h = (h - f_h + 2 * p) / s + 1
        output_layer_w = (w - f_w + 2 * p) / s + 1

        if output_layer_h % 1 != 0 or output_layer_w % 1 != 0:
            raise ValueError('Error!: hyper parameters setting is invalid!')

        return int(output_layer_h), int(output_layer_w), self.num_filters

    @staticmethod
    def zero_padding(inputs: np.ndarray, padding: int = 1) -> np.ndarray:
        canvas = None
        if len(inputs.shape) == 2:
            h, w = inputs.shape
            canvas = np.zeros((h + padding * 2, w + padding * 2))
            canvas[padding:h + padding, padding:w + padding] = inputs
        elif len(inputs.shape) == 3:
            b, h, w = inputs.shape
            canvas = np.zeros((b, h + padding * 2, w + padding * 2))
            canvas[:, padding:h + padding, padding:w + padding] = inputs
        elif len(inputs.shape) == 4:
            b, h, w, d = inputs.shape
            canvas = np.zeros((b, h + padding * 2, w + padding * 2, d))
            canvas[:, padding:h + padding, padding:w + padding] = inputs
        return canvas

    @staticmethod
    def iterate_regions(inputs: np.ndarray, kernel: (int, int) = (3, 3), stride: int = 1) -> (
            np.ndarray, int, int):
        h, w = None, None

        if len(inputs.shape) == 2:
            h, w = inputs.shape
        elif len(inputs.shape) == 3:
            _, h, w = inputs.shape
        elif len(inputs.shape) == 4:
            _, h, w, _ = inputs.shape

        h_limit = h - kernel[0] + 1
        w_limit = w - kernel[1] + 1
        img_region = None
        for i in range(0, h_limit, stride):
            for j in range(0, w_limit, stride):
                if len(inputs.shape) == 2:
                    img_region = inputs[i:(i + kernel[0]), j:(j + kernel[1])]
                elif len(inputs.shape) == 3:
                    img_region = inputs[:, i:(i + kernel[0]), j:(j + kernel[1])]
                elif len(inputs.shape) == 4:
                    img_region = inputs[:, i:(i + kernel[0]), j:(j + kernel[1]), :]
                yield img_region, i, j

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        if self.padding > 0:
            inputs = self.zero_padding(inputs, self.padding)
        self.last_input: np.ndarray = inputs

        batch_size: int = inputs.shape[0]
        output = np.zeros((batch_size,) + self.output_shape)

        if len(inputs.shape) == 3:
            for img_region, i, j in self.iterate_regions(inputs, kernel=self.kernel_size, stride=self.stride):
                for f in range(self.num_filters):
                    output[:, i, j, f] = np.sum(np.multiply(img_region, self.filters[:, :, f]))
                    # output[:, i, j, f] = np.sum(img_region * self.filters[:, :, f], axis=(1, 2))
        elif len(inputs.shape) == 4:
            #volume_depth = output.shape[-1]
            for f in range(self.num_filters):
                for img_region, i, j in self.iterate_regions(inputs, kernel=self.kernel_size, stride=self.stride):
                    output[:, i, j, f] = np.sum(np.multiply(img_region, self.filters[:, :, :, f]))
                    # output[:, i, j, d] = np.sum(img_region * self.filters[:, :, :, d], axis=(1, 2, 3))

        return output

    def backpropagation(self, d_score: np.ndarray) -> np.ndarray:
        # TODO: improve performances here

        self.d_filters = np.zeros(self.filters.shape)
        new_d_score = np.zeros(self.last_input.shape)
        batch_size = d_score.shape[0]

        # filters delta
        if len(self.input_shape) == 2:
            for b in range(batch_size):
                for img_region, i, j in self.iterate_regions(self.last_input, kernel=self.kernel_size,
                                                             stride=self.stride):
                    for f in range(self.num_filters):
                        self.d_filters[:, :, f] += np.dot(d_score[b, i, j, f], img_region[b])
                        new_d_score[b, i:i + self.kernel_size[0], j:j + self.kernel_size[1]] += self.filters[:, :, f] * \
                                                                                                d_score[b, i, j, f]

        elif len(self.input_shape) == 3:
            for b in range(batch_size):
                for img_region, i, j in self.iterate_regions(self.last_input, kernel=self.kernel_size,
                                                             stride=self.stride):
                    for f in range(self.num_filters):
                        self.d_filters[:, :, :, f] += np.dot(d_score[b, i, j, f], img_region[b])
                        new_d_score[b, i:i + self.kernel_size[0], j:j + self.kernel_size[1], :] += self.filters[:, :, :,
                                                                                                   f] * d_score[
                                                                                                       b, i, j, f]

        return new_d_score

    def update(self, learn_rate: float = 1e-0) -> None:
        self.filters = self.filters + (-learn_rate * self.d_filters)

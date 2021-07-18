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

    def __init__(self, num_filters: int, kernel_size: int = 3, padding: int = 0, stride: int = 1):
        super().__init__()
        self.num_filters: int = num_filters
        self.kernel_size: int = kernel_size
        self.padding: int = padding
        self.stride: int = stride
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
        for img_region, i, j in iterate_regions(inputs, kernel=self.kernel_size, stride=self.stride):
            for b in range(batch_size):
                depth = self.filters.shape[2] if len(self.filters.shape) == 4 else 1
                flatten_image = img_region[b].flatten()
                flatten_filters = self.filters.reshape((self.kernel_size ** 2) * depth, self.num_filters)
                prod_ = (flatten_filters.T * flatten_image).T
                output[b, i, j] = np.sum(prod_, axis=0)
        end = time.time()
        self.f_time += (end - start)
        return output

    def backpropagation(self, d_score: np.ndarray) -> np.ndarray:
        self.d_filters = np.zeros(self.filters.shape)
        new_d_score = np.zeros(self.last_input.shape)
        # filters delta
        start = time.time()
        batch_size = d_score.shape[0]
        nds = new_d_score
        for b in range(batch_size):
            for img_region, i, j in iterate_regions(self.last_input, kernel=self.kernel_size, stride=self.stride):

                """
                # Equivalent but much slower
                
                for f in range(self.num_filters):
                    if len(self.d_filters.shape) == 3:
                        self.d_filters[:, :, f] += d_score[b, i, j, f] * img_region[b]

                    elif len(self.d_filters.shape) == 4:
                        self.d_filters[:, :, :, f] += d_score[b, i, j, f] * img_region[b]
                """
                d_score_flatten = d_score[b, i, j].flatten()
                img_region_flatten = np.tile(img_region[b].flatten(), (self.num_filters, 1))
                prod_ = (img_region_flatten.T * d_score_flatten)
                prod_ = prod_.reshape(self.d_filters.shape)
                self.d_filters += prod_

                # execute this only if there is another layer to propagate
                if len(self.filters.shape) == 4:
                    # Equivalent but slower
                    """
                    for f in range(self.num_filters):
                        new_d_score[b, i:i + self.kernel_size, j:j + self.kernel_size] += \
                            d_score[b, i, j, f] * self.filters[:, :, :, f]
                    """

                    d_score_flatten = d_score[b, i, j].flatten()
                    filters_flatten = self.filters.reshape(np.prod(self.filters.shape[:-1]), self.num_filters)
                    prod_ = (filters_flatten * d_score_flatten).reshape(self.filters.shape)
                    sum_ = np.sum(prod_, axis=3)
                    nds[b, i:i + self.kernel_size, j:j + self.kernel_size] += sum_

        end = time.time()
        self.b_time += (end - start)
        return new_d_score

    def update(self, learn_rate: float = 1e-0) -> None:
        self.filters = self.filters + (-learn_rate * self.d_filters)

    def print_time(self):
        print(f"f-time:{self.f_time} | b-time:{self.b_time}")
        self.f_time = 0
        self.b_time = 0

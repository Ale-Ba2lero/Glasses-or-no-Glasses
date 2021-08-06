from model.layers.layer import Layer, LayerType
import numpy as np
from model.utility import xavier_initialization


class Conv2D(Layer):

    def __init__(self, num_filters: int, kernel_size: int = 3, padding: int = 0, stride: int = 1):
        super().__init__()
        self.num_filters: int = num_filters
        self.kernel_size: int = kernel_size
        self.padding: int = padding
        self.stride: int = stride
        self.layer_type: LayerType = LayerType.CONV

        self.filters = None
        self.biases = None
        self.d_filters = None
        self.d_biases = None
        self.last_input = None

    def setup(self, input_shape: tuple):

        self.input_shape: tuple = input_shape
        self.output_shape: tuple = self.convolution_compatibility(input_shape=input_shape,
                                                                  kernel_size=self.kernel_size,
                                                                  padding=self.padding,
                                                                  stride=self.stride) + (self.num_filters,)

        if len(input_shape) == 2:
            filters_shape = (self.kernel_size, self.kernel_size, self.num_filters)
            self.filters: np.ndarray = xavier_initialization(units=filters_shape)

        elif len(input_shape) == 3:
            filters_shape = (self.kernel_size, self.kernel_size, self.input_shape[2], self.num_filters)
            self.filters: np.ndarray = xavier_initialization(units=filters_shape)

        self.biases = np.zeros((1, self.num_filters))

    @staticmethod
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

        h_limit = (h - kernel) / stride + 1
        w_limit = (w - kernel) / stride + 1

        for i in range(0, int(h_limit), stride):
            for j in range(0, int(w_limit), stride):
                if len(inputs.shape) == 3:
                    img_region = inputs[:, i:(i + kernel), j:(j + kernel)]
                elif len(inputs.shape) == 4:
                    img_region = inputs[:, i:(i + kernel), j:(j + kernel), :]
                yield img_region, i, j

    @staticmethod
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

    @staticmethod
    def convolution_compatibility(input_shape: tuple,
                                  kernel_size: int = 3,
                                  padding: int = 0,
                                  stride: int = 1) -> (int, int):
        h, w = 0, 0
        if len(input_shape) == 2:
            h, w = input_shape
        elif len(input_shape) == 3:
            h, w, _ = input_shape

        output_layer_h = (h - kernel_size + 2 * padding) / stride + 1
        output_layer_w = (w - kernel_size + 2 * padding) / stride + 1

        if output_layer_h % 1 != 0 or output_layer_w % 1 != 0:
            raise ValueError('Error!: hyper parameters setting is invalid!')
        return int(output_layer_h), int(output_layer_w)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        self.last_input: np.ndarray = inputs
        if self.padding > 0:
            inputs = self.zero_padding(inputs, self.padding)

        batch_size: int = inputs.shape[0]
        output = np.zeros((batch_size,) + self.output_shape)

        for img_region, i, j in self.iterate_regions(inputs, kernel=self.kernel_size, stride=self.stride):
            for b in range(batch_size):
                depth = self.filters.shape[2] if len(self.filters.shape) == 4 else 1
                flatten_image = img_region[b].flatten()
                flatten_filters = self.filters.reshape((self.kernel_size ** 2) * depth, self.num_filters)
                output[b, i, j] = np.dot(flatten_filters.T, flatten_image) + self.biases
        return output

    def backpropagation(self, d_score: np.ndarray) -> np.ndarray:
        self.d_filters = np.zeros(self.filters.shape)
        self.d_biases = np.zeros(self.num_filters)
        new_d_score = np.zeros(self.last_input.shape)
        # filters delta
        batch_size = d_score.shape[0]
        for b in range(batch_size):
            self.d_biases = np.sum(d_score[b], axis=(0, 1))
            for img_region, i, j in self.iterate_regions(self.last_input, kernel=self.kernel_size, stride=self.stride):
                # Equivalent but slower
                """for f in range(self.num_filters):
                    if len(self.d_filters.shape) == 3:
                        self.d_filters[:, :, f] += d_score[b, i, j, f] * img_region[b]
                    elif len(self.d_filters.shape) == 4:
                        self.d_filters[:, :, :, f] += d_score[b, i, j, f] * img_region[b]"""
                d_score_flatten = d_score[b, i, j].flatten()
                img_region_flatten = np.tile(img_region[b].flatten(), (self.num_filters, 1))
                prod_ = (img_region_flatten.T * d_score_flatten)
                prod_ = prod_.reshape(self.d_filters.shape)
                self.d_filters += prod_

                # Execute this only if there is another layer to propagate
                if len(self.filters.shape) == 4:
                    # Equivalent but slower
                    """for f in range(self.num_filters):
                        new_d_score[b, i:i + self.kernel_size, j:j + self.kernel_size] += \
                            d_score[b, i, j, f] * self.filters[:, :, :, f]"""
                    d_score_flatten = d_score[b, i, j].flatten()
                    filters_flatten = self.filters.reshape(np.prod(self.filters.shape[:-1]), self.num_filters)
                    prod_ = (filters_flatten * d_score_flatten).reshape(self.filters.shape)
                    new_d_score[b, i:i + self.kernel_size, j:j + self.kernel_size] += np.sum(prod_, axis=3)
        return new_d_score

    def update(self, learn_rate: float = 1e-0, clip=None) -> None:
        if clip is not None:
            self.d_filters = np.clip(self.d_filters, -clip, clip)
            self.d_biases = np.clip(self.d_biases, -clip, clip)

        self.filters = self.filters + (-learn_rate * self.d_filters)
        self.biases = self.biases + (-learn_rate * self.d_biases)

    def get_deltas(self):
        return self.d_filters, self.d_biases

    def set_deltas(self, dW, db):
        self.d_filters = dW
        self.d_biases = db

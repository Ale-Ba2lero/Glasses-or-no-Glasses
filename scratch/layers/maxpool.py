import sys

from scratch.layers.layer import Layer, LayerType
import numpy as np


class MaxPool(Layer):

    def __init__(self):
        super().__init__()
        self.layer_type = LayerType.MAXPOOL
        self.last_input = None

    def setup(self, input_shape: (int, int, int)) -> None:
        self.input_shape: tuple = input_shape
        h, w, d = input_shape
        self.output_shape: (int, int, int) = (h // 2, w // 2, d)
        # print(f"Pool layer\ninput shape: {self.input_shape}\noutput shape: {self.output_shape}\n")

    @staticmethod
    def iterate_regions(inputs: np.ndarray) -> np.ndarray:
        _, h, w, _ = inputs.shape
        for i in range(h // 2):
            for j in range(w // 2):
                img_region = inputs[:, (i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2), :]
                yield img_region, i, j

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        print("F MAXPOOL")
        print(inputs.shape)
        self.last_input: np.ndarray = inputs
        batch_size, h, w, d = inputs.shape
        output = np.zeros((batch_size, h // 2, w // 2, d))

        for img_region, i, j in self.iterate_regions(inputs):
            output[:, i, j, :] = np.amax(img_region, axis=(1, 2))
        print(output.shape)
        return output

    def backpropagation(self, d_score: np.ndarray) -> np.ndarray:
        print("BP MAXPOOL")
        print(d_score.shape)
        d_input = np.zeros(self.last_input.shape)

        for img_region, i, j in self.iterate_regions(self.last_input):
            _, h, w, _ = img_region.shape
            region_max = np.amax(img_region, axis=(1, 2))

            for idx_h in range(h):
                for idx_w in range(w):
                    # If the pixel was the max value, copy the gradient to it.
                    if (img_region[:, idx_h, idx_w, :] == region_max).all:

                        d_input[:, i * 2 + idx_h, j * 2 + idx_w, :] = d_score[:, i, j, :]

        print(d_input.shape)
        return d_input

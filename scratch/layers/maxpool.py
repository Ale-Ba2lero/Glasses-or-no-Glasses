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
        self.last_input: np.ndarray = inputs
        batch_size, h, w, d = inputs.shape
        output = np.zeros((batch_size, h // 2, w // 2, d))

        for img_region, i, j in self.iterate_regions(inputs):
            output[:, i, j, :] = np.amax(img_region, axis=(1, 2))

        return output

    def backpropagation(self, d_score: np.ndarray) -> np.ndarray:
        d_input = np.zeros(self.last_input.shape)

        for img_region, i, j in self.iterate_regions(self.last_input):
            b, h, w, d = img_region.shape
            region_max = np.amax(img_region, axis=(1, 2))

            for idx_b in range(b):
                for idx_h in range(h):
                    for idx_w in range(w):
                        for idx_d in range(d):
                            # If the pixel was the max value, copy the gradient to it.
                            if img_region[idx_b, idx_h, idx_w, idx_d] == region_max[idx_b, idx_d]:
                                d_input[idx_b, i * 2 + idx_h, j * 2 + idx_w, idx_d] = d_score[idx_b, i, j, idx_d]
        return d_input

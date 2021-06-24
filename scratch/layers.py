from scratch.activations import ReLU
import numpy as np


class Dense:
    def __init__(self, num_neurons, activation=ReLU()):
        self.num_neurons = num_neurons
        self.activation = activation

    def setup(self, input_size, next_layer=None, id=None):
        self.id = id

        # multiply by 0.1 to reduce the variance of our initial values
        self.W = 0.10 * np.random.randn(input_size, self.num_neurons)

        # print (f'W{self.id}: {self.W.shape}\n{self.W}\n')
        self.b = np.zeros((1, self.num_neurons))

        #    print (f'b{self.id}: {self.b.shape}\n{self.b}\n')
        self.next_layer = next_layer

    def forward(self, input_layer):
        self.input_layer = input_layer
        output = np.dot(input_layer, self.W) + self.b
        output = self.activation.compute(output)
        return output

    def backward(self, dscore):
        # print (f'Layer-{self.id}:')
        if self.next_layer is not None:
            dscore = self.activation.backpropagation(dscore, self.next_layer.W)

        self.dW = np.dot(self.input_layer.T, dscore)
        # print (f'dW = inputs.T{self.inputs.T.shape} * dscore{dscore.shape} = {self.dW.shape}')

        self.db = np.sum(dscore, axis=0, keepdims=True)
        #    print(f'db = dscore sum{self.db.shape} = {self.db}\n')

        return dscore

    def update(self, step_size):
        self.W += -step_size * self.dW
        #    print (f'perform b{self.id} update:\n{self.b} + {-step_size} X {self.db}')
        self.b += -step_size * self.db


class Conv:

    def __init__(self, num_filters, kernel_size=(3, 3), padding=0, stride=1, activation=ReLU()):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.activation = activation

        # We divide by 9 to reduce the variance of our initial values
        self.filters = np.random.random_sample((self.num_filters,) + self.kernel_size + (3,)) / 9

    def convolution_compatibility(self, image):
        h, w, _ = image.shape
        f_h, f_w = self.kernel_size
        s = self.stride
        p = self.padding

        output_layer_h = (h - f_h + 2 * p) / s + 1
        output_layer_w = (h - f_w + 2 * p) / s + 1

        if output_layer_h % 1 != 0 or output_layer_w % 1 != 0:
            print('Error!: hyperparameters setting is invalid!')
            return None

        return output_layer_h, output_layer_w

    def zero_padding(self, image, padding=1):
        h, w, d = image.shape
        canvas = np.zeros((h + padding*2, w + padding*2, d))
        canvas[padding:h+padding, padding:w+padding] = image
        return canvas

    def iterate_regions(self, image):
        """
        Generates all possible  image regions using valid padding.
        - image is a 3d numpy array
        """
        output_size = self.convolution_compatibility(image=image)
        print (f"output_size: {output_size}")

        if output_size is not None:
            if self.padding > 0:
                image = self.zero_padding(image, self.padding)

            h, w, _ = image.shape
            h_limit = h - self.kernel_size[0] + 1
            w_limit = w - self.kernel_size[1] + 1

            for i in range(0, h_limit, self.stride):
                for j in range(0, w_limit, self.stride):
                    img_region = image[i:(i + self.kernel_size[0]), j:(j + self.kernel_size[1]), :]
                    yield img_region, i, j

    def forward(self, input):
        """
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h, w, num_filters).
        - input is a 2d numpy array
        """
        self.last_input = input

        h, w = input.shape
        output = np.zeros((h - 2, w - 2, 3, self.num_filters))

        k = 0
        for img_region, i, j in self.iterate_regions(input):
            output[i, j, :] = np.sum(img_region * self.filters, axis=(1, 2))

        return output

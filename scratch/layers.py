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

        # We multiply by 0.1 to reduce the variance of our initial values
        self.filters = np.random.random_sample((self.num_filters,) + self.kernel_size + (3,)) * 0.1

    def convolution_compatibility(self, input_image):
        h, w, _ = input_image.shape
        f_h, f_w = self.kernel_size
        s = self.stride
        p = self.padding

        output_layer_h = (h - f_h + 2 * p) / s + 1
        output_layer_w = (w - f_w + 2 * p) / s + 1

        if output_layer_h % 1 != 0 or output_layer_w % 1 != 0:
            print('Error!: hyperparameters setting is invalid!')
            return None

        return int(output_layer_h), int(output_layer_w)

    def zero_padding(self, image, padding=1):
        h, w, d = image.shape
        canvas = np.zeros((h + padding * 2, w + padding * 2, d))
        canvas[padding:h + padding, padding:w + padding] = image
        return canvas

    def iterate_regions(self, image):
        """
        Generates all possible  image regions using valid padding.
        - image is a 3d numpy array
        """
        output_size = self.convolution_compatibility(input_image=image)

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
        Returns a 3d numpy array with dimensions (h, w, 3, num_filters).
        - input is a 3d numpy array
        """
        self.last_input = input

        o_h, o_w = self.convolution_compatibility(input)
        output = []
        for f in self.filters:
            for img_region, i, j in self.iterate_regions(input):
                output.append(np.sum(img_region * f))

        output = np.array(output)
        output = output.reshape((self.num_filters, o_h, o_w))
        return output


class MaxPool:
    def iterate_regions(self, images):
        """
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2d numpy array
        """
        n, h, w = images.shape
        new_h = h // 2
        new_w = w // 2

        for m in range(n):
            for i in range(new_h):
                for j in range(new_w):
                    img_region = images[m, (i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                    yield img_region, i, j

    def forward(self, input):
        """
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array with dimensions (h / 2, w / 2, 3, num_filters).
        - input is a 3d numpy array with dimensions (h, w, 3, num_filters)
        """
        self.last_input = input

        n, h, w = input.shape
        output = np.zeros((n, h // 2, w // 2))
        for m in range(n):
            for img_region, i, j in self.iterate_regions(input):
                output[m, i, j] = np.amax(img_region)

        return output

    def backprop(self, dL_dout):
        """
        Performs a backward pass of the maxpool layer.
        Returns the loss gradient for this layer's inputs.
        - dL_dout is the loss gradient for this layer's outputs.
        """
        dL_dinput = np.zeros(self.last_input.shape)

        for img_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = img_region.shape
            amax = np.amax(img_region, axis=(0, 1))

            for i2 in range(h):
                for j2 in range(w):
                    for f2 in range(f):
                        # If the pixel was the max value, copy the gradient to it.
                        if img_region[i2, j2, f2] == amax[f2]:
                            dL_dinput[i * 2 + i2, j * 2 + j2, f2] = dL_dout[i, j, f2]

        return dL_dinput

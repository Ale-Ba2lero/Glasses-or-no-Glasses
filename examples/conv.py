
import numpy as np

class Conv3x3:

    def __init__(self, n_filters) -> None:
        self.n_filters = n_filters

        # We divide by 9 to reduce the variance of our initial values
        self.filters = np.random.randn(n_filters, 3, 3) / 9

    def iterate_regions(self, image):
        '''
        Generates all possible 3x3 image regions using valid padding.
        - image is a 2d numpy array
        '''

        h, w = image.shape

        for i in range(h - 2):
            for j in range(w - 2):
                img_region = image[i:(i + 3), j:(j + 3)]
                yield img_region, i, j

    def forward(self, input):
        '''
        Performs a forward pass of the conv layer using the given input.
        Returns a 3d numpy array with dimensions (h, w, num_filters).
        - input is a 2d numpy array
        '''
        self.last_input = input

        h, w = input.shape
        output = np.zeros((h - 2, w - 2, self.n_filters))

        k = 0
        for img_region, i, j in self.iterate_regions(input):
            output[i, j] = np.sum(img_region * self.filters, axis=(1, 2))

        return output

    def backprop(self, d_L_d_out, learn_rate):
        '''
        Performs a backward pass of the conv layer.
        - d_L_d_out is the loss gradient for this layer's outputs.
        - learn_rate is a float.
        '''
        d_L_d_filters = np.zeros(self.filters.shape)

        for img_region, i, j in self.iterate_regions(self.last_input):
            for f in range(self.n_filters):
                d_L_d_filters[f] += d_L_d_out[i, j, f] * img_region

        # Update filters
        self.filters -= learn_rate * d_L_d_filters

        #return self.d_L_d_input
        return None


class MaxPool2:
    # A Max Pooling layer using a pool size of 2.

    def iterate_regions(self, image):
        '''
        Generates non-overlapping 2x2 image regions to pool over.
        - image is a 2d numpy array
        '''
        h, w, _ = image.shape
        new_h = h // 2
        new_w = w // 2

        for i in range(new_h):
            for j in range(new_w):
                img_region = image[(i * 2):(i * 2 + 2), (j * 2):(j * 2 + 2)]
                yield img_region, i, j 

    def forward(self, input):
        '''
        Performs a forward pass of the maxpool layer using the given input.
        Returns a 3d numpy array with dimensions (h / 2, w / 2, num_filters).
        - input is a 3d numpy array with dimensions (h, w, num_filters)
        '''
        self.last_input = input

        h, w, n_filters = input.shape
        output = np.zeros((h // 2, w // 2, n_filters))

        for img_region, i, j in self.iterate_regions(input):
            output[i, j] = np.amax(img_region, axis=(0, 1))

        
        return output

    def backprop(self, d_L_d_out):
        '''
        Performs a backward pass of the maxpool layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        '''
        d_L_d_input = np.zeros(self.last_input.shape)

        for img_region, i, j in self.iterate_regions(self.last_input):
            h, w, f = img_region.shape
            amax = np.amax(img_region, axis=(0,1))

            for i2 in range(h):
                for j2 in range (w):
                    for f2 in range(f):
                        # If the pixel was the max value, copy the gradient to it.
                        if img_region[i2, j2, f2] == amax[f2]:
                            d_L_d_input[i * 2 + i2, j * 2 + j2, f2] = d_L_d_out[i, j, f2]

        return d_L_d_input

class Softmax:
    # A standard fully-connected layer with softmax activation.

    def __init__(self, input_len, nodes):
        # We divide by input_len to reduce the variance of our initial values
        self.W = np.random.randn(input_len, nodes) / input_len
        self.b = np.zeros(nodes)

    def forward(self, input):
        '''
        Performs a forward pass of the softmax layer using the given input.
        Returns a 1d numpy array containing the respective probability values.
        - input can be any array with any dimensions.
        '''
        self.last_input_shape = input.shape

        input = input.flatten()
        self.last_input = input

        input_len, nodes = self.W.shape

        totals = np.dot(input, self.W) + self.b
        self.last_totals = totals

        exp = np.exp(totals)
        return exp / np.sum(exp, axis=0)

    def backprop(self, dL_dout, learn_rate):
        '''
        Performs a backward pass of the softmax layer.
        Returns the loss gradient for this layer's inputs.
        - d_L_d_out is the loss gradient for this layer's outputs.
        '''

        # We know only 1 element of d_L_d_out will be nonzero
        for i, gradient in enumerate(dL_dout):
            if gradient == 0:
                continue

            # e^totals
            t_exp = np.exp(self.last_totals)

            # Sum of all e^totals
            S = np.sum(t_exp)

            # Gradient of out[i] against totals
            dout_dt = -t_exp[i] * t_exp / (S ** 2)
            dout_dt[i] = t_exp[i] * (S - t_exp[i]) / (S ** 2)

            # Gradients of totals against weights/biases/input
            dt_dW = self.last_input
            dt_db = 1
            dt_dinputs = self.W

            # Gradients of loss against totals
            dL_dt = gradient * dout_dt

            # Gradients of loss against weights/biases/input
            dL_dw = dt_dW[np.newaxis].T @ dL_dt[np.newaxis]
            dL_db = dL_dt * dt_db
            dL_dinputs = dt_dinputs @ dL_dt

            # Update weights / biases
            self.W -= learn_rate * dL_dw
            self.b -= learn_rate * dL_db
            return dL_dinputs.reshape(self.last_input_shape)



from scratch.loss import CategoricalCrossentropy
import numpy as np
from scratch.layers import LayerType


class Model():
    def __init__(self, layers=[], loss_function=CategoricalCrossentropy()) -> None:
        self.layers = layers
        self.loss_function = loss_function

    def train(self, X=None, y=None, epochs=1, batch_size=None, step_size=1e-0, log=False):
        self.X = X
        self.y = y
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.STEP_SIZE = step_size

        n_batches = len(X) // self.BATCH_SIZE
        extra_batch = int(len(X) % self.BATCH_SIZE > 0)

        if log >= 1:
            print(f'training set size: {len(X)}')
            print(f'epochs: {self.EPOCHS}')
            print(f'batch size: {self.BATCH_SIZE}')
            print(f'batches: {n_batches}\nextra batch: {extra_batch}\n')

        for l in range(len(self.layers)):
            self.layer_setup(self.layers[l], l, X.shape)

        for i in range(self.EPOCHS):
            for j in range(n_batches + extra_batch):
                X_batch = X[j * self.BATCH_SIZE:(j + 1) * self.BATCH_SIZE]
                y_batch = y[j * self.BATCH_SIZE:(j + 1) * self.BATCH_SIZE]

                # print("---------------------- FORWARD\n")

                # forward step
                input_layer = X_batch
                for layer in self.layers:
                    input_layer = layer.forward(input_layer)
                output = input_layer

                # calculate loss
                loss, acc, dscore = self.loss_function.calculate(output, y_batch)

                # print loss
                if i % 1000 == 0 and j == 0 and log:
                    print_loss = "{:.2}".format(loss)
                    print_acc = "{:.2%}".format(acc)
                    print(f"iteration {i}: loss {print_loss} |  acc {print_acc}")

                # print("\n---------------------- BACKWARD\n")

                # backward step
                for layer in reversed(self.layers):
                    dscore = layer.backprop(dscore=dscore)

    def layer_setup(self, layer, layer_idx, input_shape):
        if layer.layer_type == LayerType.DENSE:
            if layer_idx == 0:
                self.layers[layer_idx].setup(input_shape=input_shape, next_layer=self.layers[layer_idx + 1])
            elif layer_idx == len(self.layers) - 1:
                self.layers[layer_idx].setup(input_shape=self.layers[layer_idx - 1].output_shape)
            else:
                self.layers[layer_idx].setup(input_shape=self.layers[layer_idx - 1].output_shape,
                                             next_layer=self.layers[layer_idx + 1])
        elif layer.layer_type == LayerType.CONV:
            if layer_idx == 0:
                self.layers[layer_idx].setup(input_shape=input_shape)
            else:
                self.layers[layer_idx].setup(input_shape=self.layers[layer_idx - 1].output_shape)
        elif layer.layer_type == LayerType.MAXPOOL:
            self.layers[layer_idx].setup(input_shape=self.layers[layer_idx - 1].output_shape)
        elif layer.layer_type == LayerType.FLATTEN:
            self.layers[layer_idx].setup(input_shape=self.layers[layer_idx - 1].output_shape,
                                         next_layer=self.layers[layer_idx + 1])

    def summary(self):
        # TODO
        pass

    def evaluate(self, X_test, y_test):

        # forward step
        input_layer = X_test
        for layer in self.layers:
            input_layer = layer.forward(input_layer)
        output = input_layer

        predicted_class = np.argmax(output, axis=1)
        acc = "{:.2%}".format(np.mean(predicted_class == y_test))
        print(f'Test accuracy: {acc}')

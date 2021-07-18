from model.loss import CategoricalCrossEntropy, Loss
import numpy as np
from model.layers.layer import Layer, LayerType
from tqdm import tqdm


class Model:
    def __init__(self, layers: list, loss_function: Loss = CategoricalCrossEntropy()) -> None:
        self.layers = layers
        self.loss_function = loss_function

        self.X = None
        self.y = None
        self.EPOCHS = None
        self.BATCH_SIZE = None
        self.STEP_SIZE = None

        self.forward_time = 0
        self.backward_time = 0

    def train(self, X: np.ndarray = None, y: np.ndarray = None, epochs: int = 1, batch_size: int = None,
              step_size: float = 1e-0):
        self.X = X
        self.y = y
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        n_batches: int = len(X) // self.BATCH_SIZE
        extra_batch: int = int(len(X) % self.BATCH_SIZE > 0)

        print(f'training set size: {len(X)}')
        print(f'epochs: {self.EPOCHS}')
        print(f'batch size: {self.BATCH_SIZE}')
        print(f'batches: {n_batches}\nextra batch: {extra_batch}\n')

        for layer_idx in range(len(self.layers)):
            self.layer_setup(self.layers[layer_idx], layer_idx, X[0].shape)

        for i in tqdm(range(self.EPOCHS)):
            loss = 0
            num_correct = 0
            for j in range(n_batches + extra_batch):
                # print loss
                if j > 0 and j % 100 == 99:
                    print_loss = "{:.2}".format(loss / 100)
                    print_acc = "{:.2%}".format(num_correct / 100)
                    print(f"\niteration {j + 1}: loss {print_loss} |  acc {print_acc}")
                    loss = 0
                    num_correct = 0

                X_batch = X[j * self.BATCH_SIZE:(j + 1) * self.BATCH_SIZE]
                y_batch = y[j * self.BATCH_SIZE:(j + 1) * self.BATCH_SIZE]
                output = X_batch

                for layer in self.layers:
                    output = layer.forward(output)

                # calculate loss
                l, acc, d_score = self.loss_function.calculate(output, y_batch)
                loss += l
                num_correct += acc

                """if i % 100 == 0 and j == 0:
                    print_loss = "{:.2}".format(l)
                    print_acc = "{:.2%}".format(acc)
                    print(f"\niteration {i}: loss {print_loss} |  acc {print_acc}")
                """
                for layer in reversed(self.layers):
                    d_score = layer.backpropagation(d_score=d_score)

                # layer update
                for layer in self.layers:
                    if layer.layer_type == LayerType.CONV or layer.layer_type == LayerType.DENSE:
                        layer.update(step_size)

    def layer_setup(self, layer: Layer, layer_idx: int, input_shape: tuple):
        layer.id_ = layer_idx
        if layer.layer_type == LayerType.DENSE:
            if layer_idx == 0:
                self.layers[layer_idx].setup(input_shape=input_shape[0], next_layer=self.layers[layer_idx + 1])
            else:
                dense_input_shape = self.layers[layer_idx - 1].output_shape
                if type(self.layers[layer_idx - 1].output_shape) == tuple:
                    dense_input_shape = dense_input_shape[0]
                if layer_idx == len(self.layers) - 1:
                    self.layers[layer_idx].setup(input_shape=dense_input_shape)
                else:
                    self.layers[layer_idx].setup(input_shape=dense_input_shape, next_layer=self.layers[layer_idx + 1])
        elif layer.layer_type == LayerType.CONV and layer_idx == 0:
            self.layers[layer_idx].setup(input_shape=input_shape)
        elif layer.layer_type == LayerType.MAXPOOL or \
                layer.layer_type == LayerType.FLATTEN or \
                layer.layer_type == LayerType.RELU or \
                layer.layer_type == LayerType.SOFTMAX or \
                (layer.layer_type == LayerType.CONV and layer_idx != 0):
            self.layers[layer_idx].setup(input_shape=self.layers[layer_idx - 1].output_shape)

    def summary(self):
        pass

    def evaluate(self, X_test, y_test):
        # forward step
        output = X_test
        for layer in self.layers:
            output = layer.forward(output)

        predicted_class = np.argmax(output, axis=1)
        acc = "{:.2%}".format(np.mean(predicted_class == y_test))
        print(f'Test accuracy: {acc}')

        for layer in self.layers:
            if layer.layer_type == LayerType.CONV:
                layer.print_time()

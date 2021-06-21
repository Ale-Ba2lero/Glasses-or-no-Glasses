
from scratch.loss import CategoricalCrossentropy
import numpy as np

class Model():
    def __init__(self, layers=[], loss_function=CategoricalCrossentropy()) -> None:
        self.layers = layers
        self.loss_function = loss_function

    def train (self, X=None, y=None, epochs=1, batch_size=None, step_size=1e-0, log=False):
        self.X = X
        self.y = y
        self.EPOCHS = epochs
        self.BATCH_SIZE = batch_size
        self.STEP_SIZE = step_size

        n_batches = len(X) // self.BATCH_SIZE
        extra_batch = int(len(X) % self.BATCH_SIZE > 0)

        if log >= 1:
            print (f'training set size: {len(X)}')
            print (f'epochs: {self.EPOCHS}')
            print (f'batch size: {self.BATCH_SIZE}')
            print (f'batches: {n_batches}\nextra batch: {extra_batch}\n')

        for l in range(len(self.layers)):
            if l == 0:
                self.layers[l].setup(input_size=X.shape[1], next_layer=self.layers[l+1], id=l+1)
            elif l == len(self.layers) - 1:
                self.layers[l].setup(input_size=self.layers[l-1].num_neurons, id=l+1)
            else:
                self.layers[l].setup(input_size=self.layers[l-1].num_neurons, next_layer=self.layers[l+1], id=l+1)

        #print('\n')
                
        for i in range(self.EPOCHS):
            for j in range (n_batches + extra_batch):
                X_batch = X[j*self.BATCH_SIZE:(j+1)*self.BATCH_SIZE]
                y_batch = y[j*self.BATCH_SIZE:(j+1)*self.BATCH_SIZE]
                
                #print("---------------------- FORWARD\n")

                # forward step
                input_layer = X_batch
                for layer in self.layers:
                    input_layer = layer.forward(input_layer)
                output = input_layer

                # calculate loss
                loss, acc, dscore = self.loss_function.calculate(output, y_batch)

                # print loss
                if i % 1000 == 0 and j == 0 and log == True:
                    print_loss = "{:.2}".format(loss)
                    print_acc = "{:.2%}".format(acc)
                    print (f"iteration {i}: loss {print_loss} ---- acc {print_acc}")

                #print("\n---------------------- BACKWARD\n")

                # backward step
                for layer in reversed(self.layers):
                    dscore = layer.backward(dscore)

                # parameters update
                for layer in self.layers:
                    layer.update(self.STEP_SIZE)
                    

    def summary(self):
        #TODO
        pass

    def evaluate(self, X_test, y_test):

        # forward step
        input_layer = X_test
        for layer in self.layers:
            input_layer = layer.forward(input_layer)
        output = input_layer

        predicted_class = np.argmax(output, axis=1)
        acc = "{:.2%}".format(np.mean(predicted_class == y_test)) 
        print (f'Test accuracy: {acc}')
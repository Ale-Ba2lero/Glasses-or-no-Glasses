
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

        #print (f'W{self.id}: {self.W.shape}\n{self.W}\n')
        self.b = np.zeros((1, self.num_neurons))

        #    print (f'b{self.id}: {self.b.shape}\n{self.b}\n')
        self.next_layer = next_layer

    def forward(self, input_layer):
        self.input_layer = input_layer
        output = np.dot(input_layer, self.W) + self.b
        output = self.activation.compute(output)
        return output

    def backward(self, dscore):
        #print (f'Layer-{self.id}:')
        if self.next_layer is not None:
            dscore = self.activation.backpropagation(dscore, self.next_layer.W)
            
        self.dW = np.dot(self.input_layer.T, dscore)
        #print (f'dW = inputs.T{self.inputs.T.shape} * dscore{dscore.shape} = {self.dW.shape}')

        self.db = np.sum(dscore, axis=0, keepdims=True)
        #    print(f'db = dscore sum{self.db.shape} = {self.db}\n')

        return dscore

    def update(self, step_size):
        self.W += -step_size * self.dW
        #    print (f'perform b{self.id} update:\n{self.b} + {-step_size} X {self.db}')
        self.b += -step_size * self.db
            

from scratch.activations import ReLU
import numpy as np

class Dense:
    def __init__(self, units, activation=ReLU, use_bias=True):
        self.units = units
        self.activation = activation
        self.use_bias = use_bias
        
    def setup(self, inputs, next_layer=None, id=None):
        self.layer_inputs = inputs
        self.id = id

        self.W = 0.10 * np.random.randn(inputs, self.units)
        #print (f'W{self.id}: {self.W.shape}\n{self.W}\n')
        if self.use_bias: 
            self.b = np.zeros((1, self.units))
        #    print (f'b{self.id}: {self.b.shape}\n{self.b}\n')
        self.next_layer = next_layer

    def forward(self, inputs):
        self.inputs = inputs
        self.output = np.dot(inputs, self.W) + self.b
        self.output = self.activation.compute(self.output)

    def backward(self, dscore):
        #print (f'Layer-{self.id}:')
        if self.next_layer is not None:
            dscore = self.activation.backpropagation(dscore, self.next_layer.W)
        self.dW = np.dot(self.inputs.T, dscore)
        #print (f'dW = inputs.T{self.inputs.T.shape} * dscore{dscore.shape} = {self.dW.shape}')
        if self.use_bias: 
            self.db = np.sum(dscore, axis=0, keepdims=True)
        #    print(f'db = dscore sum{self.db.shape} = {self.db}\n')
        return dscore

    def update(self, step_size):
        self.W += -step_size * self.dW
        if self.use_bias: 
        #    print (f'perform b{self.id} update:\n{self.b} + {-step_size} X {self.db}')
            self.b += -step_size * self.db
            
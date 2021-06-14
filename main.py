
import numpy as np
import nnfs
from nnfs.datasets import spiral_data

from scratch.loss import CategoricalCrossentropy
from scratch.layers import Dense
from scratch.activations import ReLU, Softmax

nnfs.init()

# ------------------------------------ DATASET
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
h = 100 # size of hidden layer

# some hyperparameters
step_size = 1e-0
reg = 1e-3 # regularization strength

#X, y = spiral_data(samples=N, classes=K)

X = np.zeros((N*K,D)) # data matrix (each row = single example)
y = np.zeros(N*K, dtype='uint8') # class labels
for j in range(K):
  ix = range(N*j,N*(j+1))
  r = np.linspace(0.0,1,N) # radius
  t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.2 # theta
  X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
  y[ix] = j

# ------------------------------------ NN
dense1 = Dense(D, h)
relu1 = ReLU()
dense2 = Dense(h, K)
softmax = Softmax()

for i in range(10000):
    dense1.forward(X)
    relu1.forward(dense1.output)
    dense2.forward(relu1.output)
    softmax.forward(dense2.output)

    loss_function = CategoricalCrossentropy()
    loss = loss_function.calculate(softmax.output, y)
    if i % 1000 == 0:
        print (f"iteration {i}: loss {loss}")

    # 1) compute the gradient on scores
    dscore = loss_function.doutput
    # 2) backpropate the gradient to the parameters

    #       dW2 = dscore * ReLU output
    dense2.backward(dscore)

    #       backprop previous layer
    dscore = relu1.backpropagation(dscore, dense2.W)

    #       dW1 = X.T * dscore
    dense1.backward(dscore)

    # 3) perform a parameter update
    dense1.update(step_size)
    dense2.update(step_size)

#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.show()

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import nnfs
from nnfs.datasets import spiral_data

from scratch.loss import CategoricalCrossentropy
from scratch.layers import Dense
from scratch.activations import ReLU, Softmax

nnfs.init()

'''
Epoch = one cycle through the full training dataset.

Batch = number (subset) of training samples or examples in one iteration. 

The higher the batch size, the more memory space we need.

Batch Gradient Descent. Batch Size = Size of Training Set
Stochastic Gradient Descent. Batch Size = 1
Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set

The model weights will be updated after each batch
'''

#reg = 1e-3 # regularization strength

# ------------------------------------ DATASET
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes
h = 100 # size of hidden layer

X, y = spiral_data(samples=N, classes=K)

#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=12)

# ------------------------------------ HYPERPARAMETERS
STEP_SIZE = 1e-0
N_EPOCHS = 10000
BATCH_SIZE = N*K//3

# ------------------------------------ NN
dense1 = Dense(D, h)
relu1 = ReLU()
dense2 = Dense(h, K)
softmax = Softmax()

n_batches = len(X_train) // BATCH_SIZE
extra_batch = int(len(X_train) % BATCH_SIZE > 0)

print (f'# batches: {n_batches}\nextra batch: {extra_batch}')

for i in range(N_EPOCHS):
    for j in range (n_batches + extra_batch):
        X_train_batch = X_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        y_train_batch = y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE]  
        
        dense1.forward(X_train_batch)
        relu1.forward(dense1.output)
        dense2.forward(relu1.output)
        softmax.forward(dense2.output)

        loss_function = CategoricalCrossentropy()
        loss = loss_function.calculate(softmax.output, y_train_batch)
        
        if i % 1000 == 0 and j == 0:
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
        dense1.update(STEP_SIZE)
        dense2.update(STEP_SIZE)
    
# ------------------------------------ PERFORMANCE ANALYSIS
hidden_layer = np.maximum(0, np.dot(X_test, dense1.W) + dense1.b)
scores = np.dot(hidden_layer, dense2.W) + dense2.b
predicted_class = np.argmax(scores, axis=1)
print (f'Test accuracy: {np.mean(predicted_class == y_test)}')

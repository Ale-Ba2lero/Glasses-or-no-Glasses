from scratch.model import Model
import numpy as np
from sklearn.model_selection import train_test_split

import nnfs
from nnfs.datasets import spiral_data

from scratch.loss import CategoricalCrossentropy
from scratch.layers import Dense
from scratch.activations import ReLU, Softmax
from scratch.model import Model

nnfs.init()

'''
Batch Gradient Descent. Batch Size = Size of Training Set
Stochastic Gradient Descent. Batch Size = 1
Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set
'''

# ------------------------------------ DATASET
N = 100 # number of points per class
D = 2 # dimensionality
K = 3 # number of classes

X, y = spiral_data(samples=N, classes=K)

#plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=12)

# ------------------------------------ HYPER PARAMETERS
STEP_SIZE = 1e-0
N_EPOCHS = 10000
BATCH_SIZE = len(X_train)//1

# ------------------------------------ BUILD THE MODEL
nn = Model([
    Dense(30, activation=ReLU()),
    Dense(30, activation=ReLU()),
    Dense(30, activation=ReLU()),
    Dense(K, activation=Softmax())
], CategoricalCrossentropy())

# ------------------------------------ FIT THE MODEL
nn.fit(X=X_train, 
        y=y_train, 
        epochs=N_EPOCHS, 
        batch_size=BATCH_SIZE, 
        step_size=STEP_SIZE,
        log=True)

# ------------------------------------ EVALUTATE THE MODEL
nn.evaluate(X_test=X_test, y_test=y_test)

'''
n_batches = len(X_train) // BATCH_SIZE
extra_batch = int(len(X_train) % BATCH_SIZE > 0)

print (f'training set size: {len(X_train)}')
print(f'batch size: {BATCH_SIZE}')
print (f'batches: {n_batches}\nextra batch: {extra_batch}\n')
# ------------------------------------ NN

layer1 = Dense(h, activation=ReLU())
layer2 = Dense(K, activation=Softmax())

layer1.setup(inputs=X_test.shape[1], next_layer=layer2)
layer2.setup(inputs=layer1.units)

for i in range(N_EPOCHS):
    for j in range (n_batches + extra_batch):
        X_train_batch = X_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        y_train_batch = y_train[j*BATCH_SIZE:(j+1)*BATCH_SIZE]
        
        layer1.forward(X_train_batch)
        layer2.forward(layer1.output)

        loss_function = CategoricalCrossentropy()
        loss, acc = loss_function.calculate(layer2.output, y_train_batch)
        
        if i % 1000 == 0 and j == 0:
            print_loss = "{:.2}".format(loss)
            print_acc = "{:.2%}".format(acc)
            print (f"iteration {i}: loss {print_loss} ---- acc {print_acc}")
        
        # 1) compute the gradient on scores
        dscore = loss_function.doutput
        # 2) backpropate the gradient to the parameters
        #       dW2 = dscore * ReLU output
        #       backprop previous layer
        #       dW1 = X.T * dscore
        layer2.backward(dscore)
        layer1.backward(dscore)

        # 3) perform a parameter update
        layer1.update(STEP_SIZE)
        layer2.update(STEP_SIZE)
        
# ------------------------------------ PERFORMANCE ANALYSIS

hidden_layer = np.maximum(0, np.dot(X_test, layer1.W) + layer1.b)
scores = np.dot(hidden_layer, layer2.W) + layer2.b
predicted_class = np.argmax(scores, axis=1)

acc = "{:.2%}".format(np.mean(predicted_class == y_test)) 
print (f'Test accuracy: {acc}')
'''
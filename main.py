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

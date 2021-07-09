import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from scratch.loss import CategoricalCrossEntropy
from scratch.layers.dense import Dense
from scratch.activations import ReLU, Softmax, LeakyReLU
from scratch.model import Model

np.seterr(all='raise')
nnfs.init()

'''
Batch Gradient Descent. Batch Size = Size of Training Set
Stochastic Gradient Descent. Batch Size = 1
Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set
'''

# ------------------------------------ DATASET
N = 200  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes

X, y = spiral_data(samples=N, classes=K)

print("Scale values to [0;1]")
print('Min: %.3f, Max: %.3f' % (X.min(), X.max()))
X = minmax_scale(X, feature_range=(-0.5, 0.5))
print('Min: %.3f, Max: %.3f' % (X.min(), X.max()))

# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()

# ------------------------------------ SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=12)

# ------------------------------------ HYPER PARAMETERS
STEP_SIZE = 1e-1
N_EPOCHS = 10000
BATCH_SIZE = len(X_train) // 10

# ------------------------------------ BUILD THE MODEL
nn = Model([
    Dense(200, activation=ReLU()),
    Dense(100, activation=ReLU()),
    Dense(50, activation=ReLU()),
    Dense(K, activation=Softmax())
], CategoricalCrossEntropy())
# ------------------------------------ FIT THE MODEL
nn.train(X=X_train,
         y=y_train,
         epochs=N_EPOCHS,
         batch_size=BATCH_SIZE,
         step_size=STEP_SIZE)

# ------------------------------------ EVALUTATE THE MODEL
nn.evaluate(X_test=X_test, y_test=y_test)

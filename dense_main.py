import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale

import matplotlib.pyplot as plt
from model.loss import CategoricalCrossEntropy
from model.layers.dense import Dense
from model.layers.ReLU import ReLU
from model.layers.softmax import Softmax
from model.layers.leakyReLU import LeakyReLU
from model.model import Model


# np.seterr(all='raise')


def spiral_data(points, classes):
    X = np.zeros((points * classes, 2))
    y = np.zeros(points * classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points * class_number, points * (class_number + 1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number * 4, (class_number + 1) * 4, points) + np.random.randn(points) * 0.2
        X[ix] = np.c_[r * np.sin(t * 2.5), r * np.cos(t * 2.5)]
        y[ix] = class_number
    return X, y


'''
Batch Gradient Descent. Batch Size = Size of Training Set
Stochastic Gradient Descent. Batch Size = 1
Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set
'''

# ------------------------------------ DATASET
N = 200  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes

X, y = spiral_data(points=N, classes=K)

print("Scale values")
print('Min: %.3f, Max: %.3f' % (X.min(), X.max()))
X = minmax_scale(X, feature_range=(0, 1))
print('Min: %.3f, Max: %.3f' % (X.min(), X.max()))

# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()

# ------------------------------------ SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.10,
                                                    random_state=65)

# ------------------------------------ HYPER PARAMETERS
STEP_SIZE = 1e-1
N_EPOCHS = 10000
BATCH_SIZE = len(X_train) // 20

# ------------------------------------ BUILD THE MODEL
nn = Model([
    Dense(200), ReLU(),
    Dense(100), ReLU(),
    Dense(50), ReLU(),
    Dense(K), Softmax()
], CategoricalCrossEntropy())

# ------------------------------------ FIT THE MODEL
nn.train(X=X_train,
         y=y_train,
         epochs=N_EPOCHS,
         batch_size=BATCH_SIZE,
         step_size=STEP_SIZE)

# ------------------------------------ EVALUTATE THE MODEL
nn.evaluate(X_test=X_test, y_test=y_test)

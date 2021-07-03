import numpy as np
from sklearn.model_selection import train_test_split

import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt
from scratch.loss import CategoricalCrossEntropy
from scratch.layers.dense import Dense
from scratch.activations import ReLU, Softmax
from scratch.model import Model

np.seterr(all='raise')
nnfs.init()

'''
Batch Gradient Descent. Batch Size = Size of Training Set
Stochastic Gradient Descent. Batch Size = 1
Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set
'''

# ------------------------------------ DATASET
N = 100  # number of points per class
D = 2  # dimensionality
K = 3  # number of classes

X, y = spiral_data(samples=N, classes=K)

# plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
# plt.show()

# ------------------------------------ SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.20,
                                                    random_state=12)

# ------------------------------------ HYPER PARAMETERS
STEP_SIZE = 1e-0
N_EPOCHS = 9000
BATCH_SIZE = len(X_train) // 4

# ------------------------------------ BUILD THE MODEL
nn = Model([
    Dense(100, activation=ReLU()),
    Dense(10, activation=ReLU()),
    Dense(K, activation=Softmax())
], CategoricalCrossEntropy())
# ------------------------------------ FIT THE MODEL
nn.train(X=X_train,
         y=y_train,
         epochs=N_EPOCHS,
         batch_size=BATCH_SIZE,
         step_size=STEP_SIZE,
         log=True)

# ------------------------------------ EVALUTATE THE MODEL
nn.evaluate(X_test=X_test, y_test=y_test)

"""

dense_layer = Dense(100, activation=ReLU())
dense_layer2 = Dense(K, activation=Softmax())

loss_function = CategoricalCrossEntropy()

dense_layer.setup(input_shape=X_test.shape, next_layer=dense_layer2)
dense_layer2.setup(input_shape=dense_layer.output_shape)

for i in range(N_EPOCHS):
    out = dense_layer.forward(X_train)
    out = dense_layer2.forward(out)

    loss, acc, d_score = loss_function.calculate(out, y_train)

    # print loss
    if i % 1000 == 0:
        print_loss = "{:.2}".format(loss)
        print_acc = "{:.2%}".format(acc)
        print(f"iteration {i}: loss {print_loss} |  acc {print_acc}")

    d_score = dense_layer2.backpropagation(d_score)
    d_score = dense_layer.backpropagation(d_score)

    dense_layer2.update()
    dense_layer.update()
    
    """

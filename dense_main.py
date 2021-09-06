import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale
import matplotlib.pyplot as plt

from model.loss import CategoricalCrossEntropy
from model.layers.dense import Dense
from model.layers.relu import LeakyReLU
from model.layers.softmax import Softmax
from model.neural_network import NeuralNetwork


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
"""X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.1,
                                                    random_state=65)"""

# ------------------------------------ HYPER PARAMETERS
STEP_SIZE = 1e-1
N_EPOCHS = 2000
BATCH_SIZE = 32

# ------------------------------------ BUILD THE MODEL
nn = NeuralNetwork([
    Dense(200), LeakyReLU(),
    Dense(100), LeakyReLU(),
    Dense(50), LeakyReLU(),
    Dense(K), Softmax()
], CategoricalCrossEntropy())

# ------------------------------------ FIT THE MODEL
nn.train(dataset=X,
         labels=y,
         epochs=N_EPOCHS,
         batch_size=BATCH_SIZE,
         step_size=STEP_SIZE)

# ------------------------------------ EVALUATE THE MODEL
train_loss = nn.metrics.history['train_loss']
val_loss = nn.metrics.history['val_loss']
epochs = range(0, N_EPOCHS)
plt.plot(epochs, train_loss, 'g', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

print(f"train loss: {train_loss}")
print(f"val loss: {val_loss}")

train_acc = nn.metrics.history['train_acc']
val_acc = nn.metrics.history['val_acc']
epochs = range(0, N_EPOCHS)
plt.plot(epochs, train_acc, 'g', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

print(f"train acc: {train_acc}")
print(f"val acc: {val_acc}")

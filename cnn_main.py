from sklearn.model_selection import train_test_split

from scratch.layers.conv import Conv
from scratch.layers.dense import Dense
from scratch.layers.maxpool2 import MaxPool2
from scratch.layers.flatten import Flatten
from scratch.activations import ReLU, Softmax
from scratch.loss import CategoricalCrossEntropy
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from scratch.model import Model

import idx2numpy

train_images_file = 'examples/cnn/dataset/train-images.idx3-ubyte'
train_labels_file = 'examples/cnn/dataset/train-labels.idx1-ubyte'
test_images_file = 'examples/cnn/dataset/t10k-images.idx3-ubyte'
test_labels_file = 'examples/cnn/dataset/t10k-labels.idx1-ubyte'

train_images = idx2numpy.convert_from_file(train_images_file)
train_labels = idx2numpy.convert_from_file(train_labels_file)
test_images = idx2numpy.convert_from_file(test_images_file)
test_labels = idx2numpy.convert_from_file(test_labels_file)

DATASET_SIZE = 1000
TRAINING_SET_SIZE = 100

train_images = train_images[:DATASET_SIZE]
train_labels = train_labels[:DATASET_SIZE]
test_images = test_images[:TRAINING_SET_SIZE]
test_labels = test_labels[:TRAINING_SET_SIZE]

print('Min: %.3f, Max: %.3f' % (train_images.min(), train_images.max()))
train_images = train_images.astype('float32')
train_images /= 255.0
print('Min: %.3f, Max: %.3f' % (train_images.min(), train_images.max()))


X_train, X_test, y_train, y_test = train_test_split(train_images,
                                                    train_labels,
                                                    test_size=0.1,
                                                    random_state=69)
STEP_SIZE = 1e-1
N_EPOCHS = 3
BATCH_SIZE = len(X_train) // 10

# ------------------------------------ BUILD THE MODEL
nn = Model([
    Conv(num_filters=8, padding=1),
    MaxPool2(),
    Flatten(),
    Dense(10, activation=Softmax())
], CategoricalCrossEntropy())

print("Model train")
# ------------------------------------ FIT THE MODEL
nn.train(X=X_train,
         y=y_train,
         epochs=N_EPOCHS,
         batch_size=BATCH_SIZE,
         step_size=STEP_SIZE)

# ------------------------------------ EVALUTATE THE MODEL
nn.evaluate(X_test=X_test, y_test=y_test)
"""
plt.imshow(data[0])
plt.show()
"""


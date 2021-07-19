from sklearn.model_selection import train_test_split

from model.layers.conv2d import Conv2D
from model.layers.dense import Dense
from model.layers.maxpool2d import MaxPool2D
from model.layers.flatten import Flatten
from model.layers.relu import ReLU, LeakyReLU
from model.layers.softmax import Softmax
from model.layers.dropout import Dropout
from model.loss import CategoricalCrossEntropy
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from model.model import Model

import idx2numpy

"""
plt.imshow(data[0])
plt.show()
"""

train_images_file = 'examples/cnn/dataset/train-images.idx3-ubyte'
train_labels_file = 'examples/cnn/dataset/train-labels.idx1-ubyte'

train_images = idx2numpy.convert_from_file(train_images_file)
train_labels = idx2numpy.convert_from_file(train_labels_file)

DATASET_SIZE = 1000

ds_images = train_images[:DATASET_SIZE]
ds_labels = train_labels[:DATASET_SIZE]

ds_images = ds_images.astype('float32')
ds_images /= 255.0
# print('Min: %.3f, Max: %.3f' % (train_images.min(), train_images.max()))

X_train, X_test, y_train, y_test = train_test_split(ds_images,
                                                    ds_labels,
                                                    test_size=0.1,
                                                    random_state=6)
STEP_SIZE = 1e-2
N_EPOCHS = 3
BATCH_SIZE = 1  # len(X_train) // 10

# ------------------------------------ BUILD THE MODEL
nn = Model([
    Conv2D(num_filters=10, kernel_size=3, padding=1), LeakyReLU(),
    MaxPool2D(),
    Conv2D(num_filters=8, kernel_size=3, padding=1), LeakyReLU(),
    MaxPool2D(),
    Dropout(0.2),
    Flatten(),
    Dense(256), LeakyReLU(),
    Dropout(0.2),
    Dense(10), Softmax()
], CategoricalCrossEntropy())

# ------------------------------------ TRAIN THE MODEL
nn.train(X=X_train,
         y=y_train,
         epochs=N_EPOCHS,
         batch_size=BATCH_SIZE,
         step_size=STEP_SIZE)

# ------------------------------------ EVALUTATE THE MODEL
nn.evaluate(X_test=X_test, y_test=y_test)

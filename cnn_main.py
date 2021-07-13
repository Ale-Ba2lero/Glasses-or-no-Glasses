from sklearn.model_selection import train_test_split

from model.layers.conv import Conv
from model.layers.dense import Dense
from model.layers.maxpool2 import MaxPool2
from model.layers.flatten import Flatten
from model.layers.ReLU import ReLU
from model.layers.softmax import Softmax
from model.loss import CategoricalCrossEntropy
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from model.model import Model
import time

import idx2numpy

"""
plt.imshow(data[0])
plt.show()
"""

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
BATCH_SIZE = len(X_train) // len(X_train)

# ------------------------------------ BUILD THE MODEL
"""nn = Model([
    Conv(num_filters=10, padding=1), ReLU(),
    MaxPool2(),
    Flatten(),
    Dense(10), Softmax()
], CategoricalCrossEntropy())"""

forward_time = 0
backward_time = 0

conv = Conv(num_filters=10, padding=1)
relu = ReLU()
mp = MaxPool2()
flatten = Flatten()
dense = Dense(10)
softmax = Softmax()

loss_function = CategoricalCrossEntropy()

conv.setup(X_train[0].shape)
relu.setup(conv.output_shape)
mp.setup(relu.output_shape)
flatten.setup(mp.output_shape)
dense.setup(flatten.output_shape)

# ------------------------------------ TRAIN THE MODEL
"""nn.train(X=X_train,
         y=y_train,
         epochs=N_EPOCHS,
         batch_size=BATCH_SIZE,
         step_size=STEP_SIZE,
         log_freq=1)"""
n_batches: int = len(X_train) // BATCH_SIZE
extra_batch: int = int(len(X_train) % BATCH_SIZE > 0)

print(f'training set size: {len(X_train)}')
print(f'epochs: {N_EPOCHS}')
print(f'batch size: {BATCH_SIZE}')
print(f'batches: {n_batches}\nextra batch: {extra_batch}\n')

for i in tqdm(range(N_EPOCHS)):
    loss = 0
    num_correct = 0
    for j in range(n_batches + extra_batch):
        if j > 0 and j % 100 == 99:
            print_loss = "{:.2}".format(loss / 100)
            print_acc = "{:.2%}".format(num_correct / 100)
            print(f"\niteration {j + 1}: loss {print_loss} |  acc {print_acc}")

            loss = 0
            num_correct = 0

        start = time.time()
        """-----------------"""
        X_batch = X_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
        y_batch = y_train[j * BATCH_SIZE:(j + 1) * BATCH_SIZE]
        out = conv.forward(X_batch)
        out = relu.forward(out)
        out = mp.forward(out)
        out = flatten.forward(out)
        out = dense.forward(out)
        out = softmax.forward(out)
        """-----------------"""
        end = time.time()
        forward_time += (end - start)

        l, acc, d_score = loss_function.calculate(out, y_batch)
        loss += l
        num_correct += acc

        start = time.time()
        """-----------------"""
        d_score = dense.backpropagation(d_score)
        d_score = flatten.backpropagation(d_score)
        d_score = mp.backpropagation(d_score)
        d_score = relu.backpropagation(d_score)
        d_score = conv.backpropagation(d_score)
        """-----------------"""
        end = time.time()
        backward_time += (end - start)

        dense.update(STEP_SIZE)
        conv.update(STEP_SIZE)

# ------------------------------------ EVALUTATE THE MODEL
"""nn.evaluate(X_test=X_test, y_test=y_test)"""

output = conv.forward(X_test)
output = relu.forward(output)
output = mp.forward(output)
output = flatten.forward(output)
output = dense.forward(output)
output = softmax.forward(output)

predicted_class = np.argmax(output, axis=1)
accuracy = "{:.2%}".format(np.mean(predicted_class == y_test))
print(f'Test accuracy: {accuracy}')

print(forward_time)
print(backward_time)

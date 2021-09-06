from model.layers.conv2d import Conv2D
from model.layers.dense import Dense
from model.layers.maxpool2d import MaxPool2D
from model.layers.flatten import Flatten
from model.layers.relu import LeakyReLU
from model.layers.softmax import Softmax
from model.layers.dropout import Dropout
from model.loss import CategoricalCrossEntropy
import matplotlib.pyplot as plt
from model.neural_network import NeuralNetwork

import idx2numpy

train_images_file = 'dataset/train-images.idx3-ubyte'
train_labels_file = 'dataset/train-labels.idx1-ubyte'

train_images = idx2numpy.convert_from_file(train_images_file)
train_labels = idx2numpy.convert_from_file(train_labels_file)

DATASET_SIZE = 400

ds_images = train_images[:DATASET_SIZE]
ds_labels = train_labels[:DATASET_SIZE]

ds_images = ds_images.astype('float32')
ds_images /= 255.0
# print('Min: %.3f, Max: %.3f' % (train_images.min(), train_images.max()))

"""X_train, X_test, y_train, y_test = train_test_split(ds_images,
                                                    ds_labels,
                                                    test_size=0.1,
                                                    random_state=6)"""
STEP_SIZE = 1e-2
N_EPOCHS = 3
BATCH_SIZE = 32

# ------------------------------------ BUILD THE MODEL
nn = NeuralNetwork([
    Conv2D(num_filters=10, kernel_size=3, padding=1), LeakyReLU(), MaxPool2D(),
    Conv2D(num_filters=8, kernel_size=3, padding=1), LeakyReLU(), MaxPool2D(),
    Flatten(),
    Dense(128), LeakyReLU(), Dropout(0.2),
    Dense(10), Softmax()
], CategoricalCrossEntropy())

# ------------------------------------ TRAIN THE MODEL
nn.train(dataset=ds_images,
         labels=ds_labels,
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
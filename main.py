# %%
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from model.loss import CategoricalCrossEntropy
from model.layers.conv2d import Conv2D
from model.layers.dense import Dense
from model.layers.maxpool2d import MaxPool2D
from model.layers.flatten import Flatten
from model.layers.relu import ReLU, LeakyReLU
from model.layers.softmax import Softmax
from model.layers.dropout import Dropout
from model.model import Model
import matplotlib.pyplot as plt

'''
Batch Gradient Descent. Batch Size = Size of Training Set
Stochastic Gradient Descent. Batch Size = 1
Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set
'''

# ------------------------------------ DATASET

#np.seterr(all='raise')

train_path = "./dataset/train.csv"
directory = "./dataset/faces-spring-2020/faces-spring-2020/"
train_ds = pd.read_csv(train_path)

# for testing purposes we will select a subset of the whole dataset
DATASET_SIZE = 10
IMAGE_SIZE = 1024

labels = train_ds.iloc[:DATASET_SIZE, -1].to_numpy()
d = None
data = np.zeros((DATASET_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
for x in tqdm(range(DATASET_SIZE)):
    img_path = f'{directory}face-{x + 1}.png'
    img = Image.open(img_path)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)  # scale the images
    img = np.array(img)
    img = (img - np.min(img)) / np.ptp(img)
    data[x] = img


"""
width, height = d.size  
left = (width - width/3*2)/2
top = (height - height/2)/2
right = (width + width/3*2)/2
bottom = (height + height/2)/2

# Crop the center of the image
d.crop((left, top, right, bottom)).show()
"""


plt.imshow(data[0])
plt.show()

# %%
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    labels,
                                                    test_size=0.1,
                                                    random_state=48)

# ------------------------------------ HYPER PARAMETERS
STEP_SIZE = 1e-2
N_EPOCHS = 3
BATCH_SIZE = 5

# ------------------------------------ BUILD THE MODEL
nn = Model([
    Conv2D(num_filters=24, kernel_size=3, padding=1), LeakyReLU(),
    MaxPool2D(),
    Conv2D(num_filters=8, kernel_size=3, padding=1), LeakyReLU(),
    MaxPool2D(),
    Flatten(),
    Dense(10), LeakyReLU(),
    Dropout(0.2),
    Dense(2), Softmax()
], CategoricalCrossEntropy())

print("Model train")
# ------------------------------------ FIT THE MODEL
nn.train(dataset=X_train,
         labels=y_train,
         epochs=N_EPOCHS,
         batch_size=BATCH_SIZE,
         step_size=STEP_SIZE)

# ------------------------------------ EVALUTATE THE MODEL
loss_train = nn.metrics.history['train_loss']
loss_val = nn.metrics.history['val_loss']
epochs = range(0, N_EPOCHS)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

loss_train = nn.metrics.history['train_acc']
loss_val = nn.metrics.history['val_acc']
epochs = range(0, N_EPOCHS)
plt.plot(epochs, loss_train, 'g', label='Training accuracy')
plt.plot(epochs, loss_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


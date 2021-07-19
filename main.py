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
DATASET_SIZE = 200
IMAGE_SIZE = 128

labels = train_ds.iloc[:DATASET_SIZE, -1].to_numpy()

data = np.zeros((DATASET_SIZE, IMAGE_SIZE, IMAGE_SIZE, 3))
for x in tqdm(range(DATASET_SIZE)):
    img_path = f'{directory}face-{x + 1}.png'
    img = Image.open(img_path)
    img = img.resize((IMAGE_SIZE, IMAGE_SIZE), Image.ANTIALIAS)  # scale the images
    img = np.array(img)
    img = (img - np.min(img)) / np.ptp(img)
    data[x] = img

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
BATCH_SIZE = len(X_train) // 10

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
nn.train(X=X_train,
         y=y_train,
         epochs=N_EPOCHS,
         batch_size=BATCH_SIZE,
         step_size=STEP_SIZE)

# ------------------------------------ EVALUTATE THE MODEL
nn.evaluate(X_test=X_train, y_test=y_train, text="Train")
nn.evaluate(X_test=X_test, y_test=y_test, text="Test")


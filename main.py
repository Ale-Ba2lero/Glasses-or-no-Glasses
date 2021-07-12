# %%
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from model.loss import CategoricalCrossEntropy
from model.layers.conv import Conv
from model.layers.dense import Dense
from model.layers.maxpool2 import MaxPool2
from model.layers.flatten import Flatten
from model.layers.ReLU import ReLU
from model.layers.softmax import Softmax
from model.model import Model
import matplotlib.pyplot as plt

'''
Batch Gradient Descent. Batch Size = Size of Training Set
Stochastic Gradient Descent. Batch Size = 1
Mini-Batch Gradient Descent. 1 < Batch Size < Size of Training Set
'''

# ------------------------------------ DATASET

np.seterr(all='raise')

train_path = "./dataset/train.csv"
directory = "./dataset/faces-spring-2020/faces-spring-2020/"
train_ds = pd.read_csv(train_path)

# for testing purposes we will select a subset of the whole dataset
dataset_size = 15
image_size = 64

labels = train_ds.iloc[:dataset_size, -1].to_numpy()

data = np.zeros((dataset_size, image_size, image_size, 3))
for x in tqdm(range(dataset_size)):
    img_path = f'{directory}face-{x + 1}.png'
    img = Image.open(img_path)
    img = img.resize((image_size, image_size), Image.ANTIALIAS)  # scale the images
    img = np.array(img)
    img = (img - np.min(img)) / np.ptp(img)
    data[x] = img

"""plt.imshow(data[0])
plt.show()"""
# %%
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    labels,
                                                    test_size=0.1,
                                                    random_state=12)

# ------------------------------------ HYPER PARAMETERS
STEP_SIZE = 1e-1
N_EPOCHS = 100
BATCH_SIZE = len(X_train) // 1

# ------------------------------------ BUILD THE MODEL
nn = Model([
    Conv(num_filters=10, padding=1), ReLU(),
    MaxPool2(),
    Flatten(),
    Dense(10), ReLU(),
    Dense(2), Softmax()
], CategoricalCrossEntropy())

print("Model train")
# ------------------------------------ FIT THE MODEL
nn.train(X=X_train,
         y=y_train,
         epochs=N_EPOCHS,
         batch_size=BATCH_SIZE,
         step_size=STEP_SIZE, log_freq=1)

# ------------------------------------ EVALUTATE THE MODEL
nn.evaluate(X_test=X_test, y_test=y_test)

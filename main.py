from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image

from scratch.loss import CategoricalCrossEntropy
from scratch.layers.conv import Conv
from scratch.layers.dense import Dense
from scratch.layers.maxpool import MaxPool
from scratch.layers.flatten import Flatten
from scratch.activations import ReLU, Softmax
from scratch.model import Model

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
dataset_size = 50
image_size = 32

labels = train_ds.iloc[:dataset_size, -1].to_numpy()

data = np.zeros((dataset_size, image_size, image_size, 3))
for x in tqdm(range(dataset_size)):
    img_path = f'{directory}face-{x + 1}.png'
    img = Image.open(img_path)
    img = img.resize((image_size, image_size), Image.ANTIALIAS)  # scale the images
    img = np.array(img)
    img = (img - np.min(img)) / np.ptp(img)
    data[x] = img

X_train, X_test, y_train, y_test = train_test_split(data,
                                                    labels,
                                                    test_size=0.20,
                                                    random_state=12)

# ------------------------------------ HYPER PARAMETERS
STEP_SIZE = 1e-0
N_EPOCHS = 3
BATCH_SIZE = len(X_train) // 1

# ------------------------------------ BUILD THE MODEL
nn = Model([
    Conv(num_filters=5, padding=1),
    MaxPool(),
    Flatten(),
    Dense(8, activation=ReLU()),
    Dense(2, activation=Softmax())
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

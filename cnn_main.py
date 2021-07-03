# %%
from scratch.layers.conv import Conv
from scratch.layers.dense import Dense
from scratch.layers.maxpool import MaxPool
from scratch.layers.flatten import Flatten
from scratch.activations import ReLU, Softmax
from scratch.loss import CategoricalCrossEntropy
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt


# %%
train_path = "./dataset/train.csv"
directory = "./dataset/faces-spring-2020/faces-spring-2020/"
train_ds = pd.read_csv(train_path)

# for testing purposes we will select a subset of the whole dataset
dataset_size = 100
image_size = 64

labels = train_ds.iloc[:dataset_size, -1].to_numpy()

# %%
data = np.zeros((dataset_size, image_size, image_size, 3))
for x in tqdm(range(dataset_size)):
    img_path = f'{directory}face-{x + 1}.png'
    img = Image.open(img_path)
    img = img.resize((image_size, image_size), Image.ANTIALIAS)  # scale the images
    img = np.array(img)
    img = (img - np.min(img)) / np.ptp(img)
    data[x] = img

conv_l = Conv(num_filters=4, kernel_size=(3, 3), padding=1, stride=1)
pool_l = MaxPool()
pool_l2 = MaxPool()
flatten_l = Flatten()
dense_l = Dense(num_neurons=8, activation=ReLU())
dense_softmax = Dense(num_neurons=2, activation=Softmax())

loss_function = CategoricalCrossEntropy()

conv_l.setup(input_shape=data.shape)
pool_l.setup(input_shape=conv_l.output_shape)
pool_l2.setup(input_shape=pool_l.output_shape)
flatten_l.setup(input_shape=pool_l2.output_shape, next_layer=dense_l)
dense_l.setup(input_shape=flatten_l.output_shape, next_layer=dense_softmax)
dense_softmax.setup(input_shape=dense_l.output_shape)

for i in range(100):
    out = conv_l.forward(data)
    out = pool_l.forward(out)
    out = pool_l2.forward(out)
    out = flatten_l.forward(out)
    out = dense_l.forward(out)
    out = dense_softmax.forward(out)

    loss, acc, dscore = loss_function.calculate(out, labels)

    print_loss = "{:.2}".format(loss)
    print_acc = "{:.2%}".format(acc)
    print(f"loss:{print_loss} | acc:{print_acc}\n")

    dscore = dense_softmax.backpropagation(dscore)
    dscore = dense_l.backpropagation(dscore)
    dscore = flatten_l.backpropagation(dscore)
    dscore = pool_l2.backpropagation(dscore)
    dscore = pool_l.backpropagation(dscore)
    conv_l.backpropagation(dscore)

    dense_softmax.update()
    dense_l.update()
    conv_l.update()

"""
plt.imshow(data[0])
plt.show()
"""

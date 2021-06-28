# %%
from scratch.layers import Conv, MaxPool, Dense, Flatten
from scratch.activations import ReLU, Softmax
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
dataset_size = 5
image_size = 64

# %%
data = np.zeros((dataset_size, image_size, image_size, 3))
for x in tqdm(range(dataset_size)):
    img_path = f'{directory}face-{x + 1}.png'
    img = Image.open(img_path)
    img = img.resize((image_size, image_size), Image.ANTIALIAS)  # scale the images
    img = np.array(img)
    img = (img - np.min(img)) / np.ptp(img)
    data[x] = img

"""
plt.imshow(data[0])
plt.show()
"""
conv_l = Conv(num_filters=3, kernel_size=(3, 3), padding=1, stride=1, log=False)
pool_l = MaxPool()
pool_l2 = MaxPool()
flatten_l = Flatten()
dense_l = Dense(num_neurons=5, activation=ReLU())
dense_softmax = Dense(num_neurons=2, activation=Softmax())

conv_l.setup(input_size=data.shape)
pool_l.setup(input_size=conv_l.output_size)
pool_l2.setup(input_size=pool_l.output_size)
flatten_l.setup(input_size=pool_l2.output_size)
dense_l.setup(input_size=flatten_l.output_size, next_layer=dense_softmax)
dense_softmax.setup(input_size=dense_l.num_neurons, next_layer=dense_softmax)

out = conv_l.forward(data)
out = pool_l.forward(out)
out = pool_l2.forward(out)
out = flatten_l.forward(out)
out = dense_l.forward(out)
out = dense_softmax.forward(out)

print(out)
"""
comparison = out[1, :, :, 0] == out[1, :, :, 1]
equal_arrays = comparison.all()
print(equal_arrays)
"""

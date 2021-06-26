# %%
from scratch.layers import Conv, MaxPool, Dense
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
dataset_size = 2
image_size = 128

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
conv_layer1 = Conv(num_filters=5, kernel_size=(3, 3), padding=1, stride=1, log=False)
conv_layer2 = Conv(num_filters=10, kernel_size=(3, 3), padding=1, stride=1, log=False)
"""pool_layer1 = MaxPool()
pool_layer2 = MaxPool()"""

conv_layer1.setup(data.shape)
conv_layer2.setup(conv_layer1.output_size)
"""pool_layer1.setup(conv_layer1.output_size)
conv_layer2.setup(pool_layer1.output_size)
pool_layer2.setup(conv_layer2.output_size)"""

out = conv_layer1.forward(data)
plt.imshow(out[0, :, :, 0])
plt.show()
plt.imshow(out[1, :, :, 0])
plt.show()

out = conv_layer2.forward(out)
"""out = pool_layer1.forward(out)
print('pool1 done')
out = conv_layer2.forward(out)
print('conv2 done')
out = pool_layer2.forward(out)
print('pool2 done')"""

plt.imshow(out[0, :, :, 0])
plt.show()
plt.imshow(out[1, :, :, 0])
plt.show()

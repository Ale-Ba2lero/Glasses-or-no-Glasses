# %%
from scratch.layers import Conv, MaxPool
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
dataset_size = 1
image_size = 28

# %%
data = []
for x in tqdm(range(dataset_size)):
    img_path = f'{directory}face-{x + 1}.png'
    img = Image.open(img_path)
    img = img.resize((image_size, image_size), Image.ANTIALIAS)  # scale the images
    img = np.array(img)
    img = (img - np.min(img)) / np.ptp(img)
    data.append(img)

conv_layer = Conv(num_filters=5, kernel_size=(3, 3), padding=1, stride=1)
pool_layer = MaxPool()
out = conv_layer.forward(data[0])


# --------------------------------------- ORIGINAL IMAGE
print('Original Image')
print(data[0].shape)
plt.imshow(data[0])
plt.show()

# --------------------------------------- FILTER
print('Filters')
print(conv_layer.filters.shape)
plt.imshow(conv_layer.filters[0])
plt.show()

# --------------------------------------- FILTERED IMAGE
print('Filtered Image')
print(out[0].shape)
plt.imshow(out[0])
plt.show()


# --------------------------------------- POOLED IMAGE
print('Pooled Image')
out = pool_layer.forward(out)
print(out.shape)
plt.imshow(out[0])
plt.show()
"""
img_regions = []
for img_region, i, j in layer.iterate_regions(data[0]):
    img_regions.append(img_region)

print(layer.zero_padding(data[0], 1).shape)
plt.imshow(layer.zero_padding(data[0], 1))
plt.show()
"""

# %%
"""
def convolution_compatibility(input_image):
    h, w, _ = input_image.shape
    f_h, f_w = 2, 2
    s = 1
    p = 0

    output_layer_h = (h - f_h + 2 * p) / s + 1
    output_layer_w = (w - f_w + 2 * p) / s + 1

    return int(output_layer_h), int(output_layer_w)


def iterate_regions(image):
    output_size = convolution_compatibility(input_image=image)

    if output_size is not None:
        h, w, _ = image.shape
        h_limit = h - 2 + 1
        w_limit = w - 2 + 1

        for i in range(0, h_limit, 1):
            for j in range(0, w_limit, 1):
                img_region = image[i:(i + 2), j:(j + 2), :]
                yield img_region, i, j


def forward(inp, ker):
    o_h, o_w = convolution_compatibility(inp)
    output = []
    for k in ker:
        for img_region, i, j in iterate_regions(inp):
            output.append(np.sum(img_region * k))

    output = np.array(output)
    output = output.reshape(2, o_h, o_w)
    return output


ker = np.array([[[[1, 5, 9],
                 [2, 6, 10]],
                [[3, 7, 11],
                 [4, 8, 12]]],
               [[[13, 17, 21],
                 [14, 18, 22]],
                [[15, 19, 23],
                 [16, 20, 24]]]]
               )

inp = np.array([[[1., 17., 33.],
                 [2., 18., 34.],
                 [3., 19., 35.],
                 [4., 20., 36.]],

                [[5., 21., 37.],
                 [6., 22., 38.],
                 [7., 23., 39.],
                 [8., 24., 40.]],

                [[9., 25., 41.],
                 [10., 26., 42.],
                 [11, 27., 43.],
                 [12., 28., 44]],

                [[13., 29., 45.],
                 [14., 30., 46.],
                 [15, 31., 47.],
                 [16., 32., 48]]
                ])

out = forward(inp, ker)
print(out)

"""

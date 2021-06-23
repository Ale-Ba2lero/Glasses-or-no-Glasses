
#%%
from scratch.layers import Conv
import numpy as np
import pandas as pd
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import math

train_path = "./dataset/train.csv"
directory = "./dataset/faces-spring-2020/faces-spring-2020/"
train_ds = pd.read_csv(train_path)

# for testing purposes we will select a subset of the whole dataset
dataset_size = 1
image_size = 28

data = []
for x in tqdm(range(dataset_size)):
  img_path = f'{directory}face-{x + 1}.png'
  img = Image.open(img_path)
  img = img.resize((image_size, image_size), Image.ANTIALIAS)       # scale the images
  img = np.array(img)
  img = (img - np.min(img))/np.ptp(img)
  data.append(img)


#plt.imshow(data[0])

layer = Conv(1, (3, 3), 0)

img_regions = []
l = 0
for img_region, i, j in layer.iterate_regions(data[0]):
  img_regions.append(img_region)
    

# %%
plt.imshow(img_regions[450])

# %%

import numpy as np


# Xavier initialization
def he_initialization(units, shape=None):
    if shape is None:
        shape = units
    stddev = np.sqrt(2 / np.prod(units))
    return np.random.normal(loc=0, scale=stddev, size=shape)

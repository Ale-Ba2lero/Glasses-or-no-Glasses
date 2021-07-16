# %%
"""import numpy as np


def compute1(inputs: np.ndarray) -> np.ndarray:
    # subtract the max (prevent overflow)
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    return probabilities


def compute2(inputs: np.ndarray) -> np.ndarray:
    log = np.log(np.sum(np.exp(inputs), axis=1))
    sub = np.subtract(inputs.T, log).T
    exp = np.exp(sub)
    return exp


def compute3(inputs: np.ndarray) -> np.ndarray:
    max = np.max(inputs, axis=1, keepdims=True)
    return max + np.log(np.sum(np.exp(inputs - max)))


def log_sum_exp(inputs: np.ndarray) -> np.ndarray:
    c = np.max(inputs, axis=1, keepdims=True)
    exp_ = np.exp(inputs - c)
    sum_ = np.sum(exp_, axis=1, keepdims=True)
    log_ = np.log(sum_)
    return c + log_


inp = np.array([[-1000, 0.4, 0.6],
                [1000, 1000, 1000],
                [0.3, 0.2, 0.2],
                [0.9, 0.7, 0.8]])

print(compute1(inp))
print(compute2(inp))
print(np.exp(inp - log_sum_exp(inp)))"""

# %%

import numpy as np

filters = np.array([[[0, 1, 0],
                     [0, 1, 0],
                     [0, 1, 0]],
                    [[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]]])

img = np.array([[[0.5, 0.7, 0.2],
                 [0.4, 0.7, 0.2],
                 [0.4, 0.3, 0.2]]])

out = np.zeros((2, 1))

flat_filters = filters.reshape(2, 9)
flat_image = img.flatten()
pr = flat_image * flat_filters

pr = pr.reshape(2, 3, 3)
print(pr)
out = np.sum(pr, axis=(1, 2))

print(out)

# %%
import numpy as np


def backpropagation(d_score):
    new_d_score = [0.5 if x < 0 else 1 for x in d_score]
    return new_d_score


d = np.array([0.4, 1.2, -3, 0.9, -1, 0.2, 3])
o = np.array([-1, 3, 2, -9, -1, 4, -3])

print(backpropagation(d))
d[o <= 0] = 0
print(d)

# %%
import numpy as np


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
print(np.exp(inp - log_sum_exp(inp)))




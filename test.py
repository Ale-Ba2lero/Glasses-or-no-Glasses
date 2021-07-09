# %%
import numpy as np

# %%
A = [[0.4, 0.2, 0.5],
     [0.3, 0.7, 0.2],
     [0.1, 0.2, 0.5],
     [0.4, 0.3, 0.8]]

B = np.sum(A, axis=0, keepdims=True)
C = np.sum(A, axis=1, keepdims=True)
print(C)

# %%

a = np.array([1, 2, 3])
print(type(a))

# %%

A = [[[2, 2, 2],
      [2, 2, 2],
      [2, 2, 2]],
     [[3, 3, 3],
      [3, 3, 3],
      [3, 3, 3]]]

B = [[2, 2, 2],
     [2, 2, 2],
     [2, 2, 2]]

A = np.array(A)
B = np.array(B)

print(f"A shape: {A.shape}")
print(f"B shape: {B.shape}")

C = A * B
D = np.sum(C, axis=(1, 2), keepdims=True)
print(f"C shape: {C.shape}")
print(C, "\n")
print(f"D shape: {D.shape}")
print(D)

# %%

data = [[5, -5, 5],
        [8, -7, 3],
        [7, -8, 1],
        [1, -2, 1],
        [6, -6, 6],
        [4, -6, 2],
        [5, -4, 3],
        [6, -3, 6],
        [7, -9, 7],
        [2, -8, 5]]
axis = 0
print(data - np.max(data, axis=1, keepdims=True))
print(np.exp(data - np.max(data, axis=1, keepdims=True)))
# data = data - np.max(data, axis=1, keepdims=True)

# %%

data = [[[[1, 2, 3], [4, 5, 6]],
         [[7, 8, 9], [0, 11, 0]]],

        [[[13, 14, 15], [16, 17, 18]],
         [[19, 20, 21], [22, 23, 24]]],

        [[[25, 26, 27], [28, 29, 30]],
         [[31, 32, 33], [34, 35, 36]]],

        [[[37, 38, 39], [40, 41, 42]],
         [[43, 44, 45], [46, 47, 48]]]]
data = np.array(data)

print(np.amax(data, axis=(1, 2)))

# %%
z = [2, 3, -2, 4, 1, -2]
print(np.greater(z, 0).astype(int))

# %%

a = [[1, 2, 3, 4, 5],
     [6, 7, 8, 9, 10],
     [11, 12, 13, 14, 15],
     [16, 17, 18, 19, 20],
     [21, 22, 23, 24, 25]]

a = np.array(a)

print(a[1:1 + 3, 1:1 + 3])


# %%

def compute2(inputs: np.ndarray) -> np.ndarray:
    log = np.log(np.sum(np.exp(inputs), axis=1))
    sub = np.subtract(inputs.T, log).T
    exp = np.exp(sub)
    return exp


def compute1(inputs: np.ndarray) -> np.ndarray:
    # subtract the max (prevent overflow)
    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    # probabilities = np.exp(inputs - np.log(np.sum(inputs, axis=1)))
    return probabilities


inp = [[0.6, 0.4, 0.6],
       [0.4, 0.1, 0.6],
       [0.3, 0.2, 0.2],
       [0.9, 0.7, 0.8]]

inp = np.array(inp)

print(compute1(inp))
print(compute2(inp))

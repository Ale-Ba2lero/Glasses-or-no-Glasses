# %%

import numpy as np

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
D = np.sum(C, axis=(1,2), keepdims=True)
print(f"C shape: {C.shape}")
print(C, "\n")
print(f"D shape: {D.shape}")
print(D)

#%%

import numpy as np

A = [[0.4, 0.2, 0.5],
     [0.3, 0.7, 0.2],
     [0.1, 0.2, 0.5],
     [0.4, 0.3, 0.8]]

B = np.sum(A, axis=0, keepdims=True)
C = np.sum(A, axis=1, keepdims=True)
print(C)

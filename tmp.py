import numpy as np
a = np.array([1, 0, 3])
print(a.shape)
b = np.zeros((a.size, a.max() + 1))
print(b.shape)
print(np.arange(a.size), a)
b[np.arange(a.size), a] = 1
print(b)

import numpy as np

np.random.seed(0)
a = np.random.randint(0, 5, size=(2, 6))
print(a)
print(a.shape)

a = a.reshape(2, 2, 3)
print(a)
print(a.shape)

a = np.sum(a, axis=1)
print(a)
print(a.shape)

a = np.tile(a, (1, 2))
print(a)
print(a.shape)

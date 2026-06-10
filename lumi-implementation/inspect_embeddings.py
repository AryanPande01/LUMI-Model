# inspect_embeddings.py

import numpy as np

E = np.load("embeddings.npy")

print(E.shape)

print("min", E.min())
print("max", E.max())
print("mean", E.mean())
print("std", E.std())

print(E[:5,:5])
# inspect_cluster.py

import numpy as np

M = np.load("cluster_matrix.npy")

print("Shape:", M.shape)
print("NonZero:", np.count_nonzero(M))
print("Density:", np.count_nonzero(M)/(542*542))

row_counts = M.sum(axis=1)

print("Min degree:", row_counts.min())
print("Max degree:", row_counts.max())
print("Mean degree:", row_counts.mean())
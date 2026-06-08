# count_cluster_edges.py

import numpy as np

cluster = np.load(
    "cluster_matrix.npy"
)

print("Shape:", cluster.shape)

print(
    "Non Zero:",
    (cluster != 0).sum()
)

print(
    "Density:",
    (cluster != 0).sum() /
    cluster.size
)
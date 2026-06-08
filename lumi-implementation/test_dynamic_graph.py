import numpy as np
import torch

from dynamic_graph import DynamicGraphBuilder

cluster_matrix = np.load(
    "cluster_matrix.npy"
)

cluster_matrix = torch.tensor(
    cluster_matrix,
    dtype=torch.float32
)

B = 2
N = 542

dummy_features = torch.randn(
    B,
    N,
    1
)

builder = DynamicGraphBuilder()

At = builder(
    dummy_features,
    cluster_matrix
)

print(
    "Dynamic Graph Shape:",
    At.shape
)

print(
    "Attention Range:",
    At.min().item(),
    At.max().item()
)
from dataset import StockDataset
from dynamic_graph import DynamicGraphBuilder
from input_projection import InputProjection

import numpy as np
import torch

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=100
)

x, _ = dataset[0]

x = x.unsqueeze(0)

projection = InputProjection(
    input_dim=5,
    hidden_dim=16
)

x = projection(x)

latest = x[:, -1]

cluster = torch.tensor(
    np.load("cluster_matrix.npy")
).float()

builder = DynamicGraphBuilder(
    feature_dim=16,
    hidden_dim=32
)

graph = builder(
    latest,
    cluster
)

print("mean =", graph.mean().item())
print("std =", graph.std().item())
print("min =", graph.min().item())
print("max =", graph.max().item())
# inspect_dynamic_graph.py

import torch
import numpy as np

from dynamic_graph import DynamicGraphBuilder
from dataset import StockDataset

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=20
)

x, y = dataset[0]

xt = x[-1]          # [542,5]

cluster = np.load("cluster_matrix.npy")

cluster = torch.tensor(
    cluster,
    dtype=torch.float32
)

layer = DynamicGraphBuilder(
    feature_dim=16,
    hidden_dim=32
)

# IMPORTANT
# if your model uses InputProjection
# import it and project first

from input_projection import InputProjection

proj = InputProjection(
    input_dim=5,
    hidden_dim=16
)

xt = proj(
    xt.unsqueeze(0)
).squeeze(0)

A = layer(
    xt.unsqueeze(0),
    cluster
)

print("Shape:", A.shape)

print("Min:", A.min().item())
print("Max:", A.max().item())
print("Mean:", A.mean().item())

print("Std:", A.std().item())
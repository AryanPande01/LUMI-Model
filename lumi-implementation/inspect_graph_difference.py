# inspect_graph_difference.py

from dataset import StockDataset
from model import LUMI

import numpy as np
import torch

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=100
)

x, _ = dataset[0]
x = x.unsqueeze(0)

model = LUMI()

with torch.no_grad():

    x = model.input_projection(x)

    latest = x[:, -1]

    dynamic = model.dynamic_graph(
        latest,
        torch.tensor(
            np.load("cluster_matrix.npy")
        ).float()
    )

print("dynamic mean", dynamic.mean().item())
print("dynamic std", dynamic.std().item())
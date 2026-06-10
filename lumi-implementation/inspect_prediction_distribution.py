# inspect_prediction_distribution.py

from dataset import StockDataset
from model import LUMI

import torch
import numpy as np

device = "cpu"

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=100
)

x, y = dataset[0]

x = x.unsqueeze(0)

model = LUMI(
    num_nodes=542,
    hidden_dim=16,
    horizon=12
)

with torch.no_grad():

    pred = model(
        x,
        torch.tensor(np.load("cluster_matrix.npy")).float(),
        torch.eye(542),
        torch.eye(542)
    )

print("pred min =", pred.min().item())
print("pred max =", pred.max().item())
print("pred mean =", pred.mean().item())
print("pred std =", pred.std().item())
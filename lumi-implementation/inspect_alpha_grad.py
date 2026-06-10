# inspect_alpha_grad.py

from dataset import StockDataset
from model import LUMI

import torch
import numpy as np

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=100
)

x, y = dataset[0]

x = x.unsqueeze(0)
y = y.unsqueeze(0)

model = LUMI()

cluster = torch.tensor(
    np.load("cluster_matrix.npy"),
    dtype=torch.float32
)

industry = torch.eye(542)
wiki = torch.eye(542)

pred = model(
    x,
    cluster,
    industry,
    wiki
)

loss = torch.nn.L1Loss()(pred, y)

loss.backward()

print("alpha =", model.semantic_graph.alpha.item())
print("grad  =", model.semantic_graph.alpha.grad)
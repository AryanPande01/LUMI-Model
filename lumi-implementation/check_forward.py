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

model = LUMI()

cluster = torch.tensor(
    np.load("cluster_matrix.npy"),
    dtype=torch.float32
)

industry = torch.eye(542)
wiki = torch.eye(542)

with torch.no_grad():

    out = model(
        x,
        cluster,
        industry,
        wiki
    )

print(out.shape)
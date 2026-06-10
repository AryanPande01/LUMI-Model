# inspect_prediction_attention.py

from dataset import StockDataset
from model import LUMI

import torch
import numpy as np

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=100
)

x, _ = dataset[0]

x = x.unsqueeze(0)

model = LUMI()

with torch.no_grad():

    x = model.input_projection(x)

    H = model.temporal_encoder(
        x,
        torch.eye(542)
    )

    H = model.temporal_attention(H)

    out = model.prediction_attention(H)

print(out.mean().item())
print(out.std().item())
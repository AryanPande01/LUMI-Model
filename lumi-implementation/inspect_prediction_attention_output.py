# inspect_prediction_attention_output.py

from dataset import StockDataset
from model import LUMI

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

    H = model.temporal_encoder(
        x,
        torch.eye(542)
    )

    H = model.temporal_attention(H)

    H = model.prediction_attention(H)

print("shape =", H.shape)
print("mean  =", H.mean().item())
print("std   =", H.std().item())
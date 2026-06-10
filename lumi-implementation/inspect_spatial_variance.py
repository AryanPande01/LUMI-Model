# inspect_spatial_variance.py
from dataset import StockDataset
from model import LUMI
import torch

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=100
)

x,_ = dataset[0]
x = x.unsqueeze(0)

model = LUMI()

with torch.no_grad():

    projected = model.input_projection(x)

    print(
        "before spatial:",
        projected.var(dim=1).mean().item()
    )

    encoded = model.temporal_encoder(
        projected,
        torch.eye(542)
    )

    print(
        "after spatial:",
        encoded.var(dim=1).mean().item()
    )
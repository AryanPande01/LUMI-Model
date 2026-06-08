# inspect_predictions.py

import torch
import numpy as np

from dataset import StockDataset
from model import LUMIStage1

dataset = StockDataset(
    "data/LSE/data/price_data.csv",
    "data/LSE/data/gt.csv",
    lookback=60
)

x, y = dataset[0]

cluster_matrix = np.load(
    "cluster_matrix.npy"
)

cluster_matrix = torch.tensor(
    cluster_matrix,
    dtype=torch.float32
)

model = LUMIStage1()

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location="cpu"
    )
)

model.eval()

with torch.no_grad():

    pred = model(
        x.unsqueeze(0),
        cluster_matrix
    )

print("Prediction Mean:", pred.mean().item())
print("Prediction Std :", pred.std().item())

print("Target Mean    :", y.mean().item())
print("Target Std     :", y.std().item())
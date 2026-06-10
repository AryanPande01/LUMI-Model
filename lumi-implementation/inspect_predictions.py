# inspect_predictions.py

import torch
import numpy as np

from dataset import StockDataset
from model import LUMI
from static_graph_loader import load_static_graphs

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=20
)

x, y = dataset[0]

x = x.unsqueeze(0).to(device)

cluster_matrix = np.load(
    "cluster_matrix.npy"
)

cluster_matrix = torch.tensor(
    cluster_matrix,
    dtype=torch.float32
).to(device)

industry_graph, wiki_graph = load_static_graphs(device)

model = LUMI(
    num_nodes=542,
    hidden_dim=16,
    horizon=12
).to(device)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

model.eval()

with torch.no_grad():

    pred = model(
        x,
        cluster_matrix,
        industry_graph,
        wiki_graph
    )

print("PRED")
print("min :", pred.min().item())
print("max :", pred.max().item())
print("mean:", pred.mean().item())
print("std :", pred.std().item())

print()

print("TARGET")
print("min :", y.min().item())
print("max :", y.max().item())
print("mean:", y.mean().item())
print("std :", y.std().item())
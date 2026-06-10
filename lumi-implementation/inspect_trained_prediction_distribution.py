from dataset import StockDataset
from model import LUMI
from static_graph_loader import load_static_graphs

import numpy as np
import torch

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

model.load_state_dict(
    torch.load(
        "final_model.pth",
        map_location=device
    )
)

model.eval()

cluster_matrix = torch.tensor(
    np.load("cluster_matrix.npy"),
    dtype=torch.float32
)

industry_graph, wiki_graph = load_static_graphs(device)

with torch.no_grad():

    pred = model(
        x,
        cluster_matrix,
        industry_graph,
        wiki_graph
    )

print("pred mean =", pred.mean().item())
print("pred std =", pred.std().item())
print("target mean =", y.mean().item())
print("target std =", y.std().item())
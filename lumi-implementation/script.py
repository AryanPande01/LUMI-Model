import numpy as np
import torch

from dataset import StockDataset
from model import LUMI
from static_graph_loader import load_static_graphs

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=60
)

x, y = dataset[0]

x = x.unsqueeze(0)

cluster_matrix = np.load(
    "cluster_matrix.npy"
)

cluster_matrix = torch.tensor(
    cluster_matrix,
    dtype=torch.float32
)

industry_graph, wiki_graph = (
    load_static_graphs(
        torch.device("cpu")
    )
)

model = LUMI(
    num_nodes=542,
    hidden_dim=16,
    horizon=12
)

out = model(
    x,
    cluster_matrix,
    industry_graph,
    wiki_graph
)

print("X:", x.shape)
print("Y:", y.shape)
print("OUT:", out.shape)
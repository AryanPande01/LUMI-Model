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

cluster_matrix = torch.tensor(
    np.load("cluster_matrix.npy"),
    dtype=torch.float32
)

industry_graph, wiki_graph = load_static_graphs(device)

model = LUMI()

with torch.no_grad():

    x = model.input_projection(x)

    semantic_graph = model.semantic_graph(
        industry_graph,
        wiki_graph
    )

    H = model.temporal_encoder(
        x,
        semantic_graph
    )

    print(
        "shape:",
        H.shape
    )

    print(
        "variance across time:",
        H.var(dim=1).mean().item()
    )
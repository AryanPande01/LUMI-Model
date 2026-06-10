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

    B, T, N, D = H.shape

    H2 = H.permute(
        0, 2, 1, 3
    ).reshape(
        B * N,
        T,
        D
    )

    Q = model.temporal_attention.query(H2)
    K = model.temporal_attention.key(H2)

    scores = torch.matmul(
        Q,
        K.transpose(-1, -2)
    )

    scores = scores / (D ** 0.5)

    alpha = torch.softmax(
        scores,
        dim=-1
    )

    print(
        "attention std:",
        alpha.std().item()
    )

    print(
        "attention mean:",
        alpha.mean().item()
    )
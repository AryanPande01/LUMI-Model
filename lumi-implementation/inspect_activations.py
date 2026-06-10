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

industry_graph, wiki_graph = load_static_graphs(device)

with torch.no_grad():

    x = model.input_projection(x)

    print(
        "after input projection:",
        x.std().item()
    )

    semantic_graph = model.semantic_graph(
        industry_graph,
        wiki_graph
    )

    short_seq, long_seq = (
        model.long_short_builder(x)
    )

    temp = model.temporal_encoder(
        short_seq,
        semantic_graph
    )

    print(
        "after temporal encoder:",
        temp.std().item()
    )

    temp = model.temporal_attention(
        temp
    )

    print(
        "after temporal attention:",
        temp.std().item()
    )

    temp = model.prediction_attention(
        temp
    )

    print(
        "after prediction attention:",
        temp.std().item()
    )
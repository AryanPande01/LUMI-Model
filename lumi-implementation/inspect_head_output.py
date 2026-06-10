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

x, _ = dataset[0]
x = x.unsqueeze(0)

cluster_matrix = torch.tensor(
    np.load("cluster_matrix.npy"),
    dtype=torch.float32
)

industry_graph, wiki_graph = load_static_graphs(device)

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

with torch.no_grad():

    x = model.input_projection(x)

    semantic_graph = model.semantic_graph(
        industry_graph,
        wiki_graph
    )

    short_seq, long_seq = (
        model.long_short_builder(x)
    )

    S1 = model.temporal_encoder(
        short_seq,
        semantic_graph
    )

    S1 = model.temporal_attention(S1)

    S1 = model.prediction_attention(S1)

    S1 = model.decoder(
        S1,
        semantic_graph
    )

    y = model.head(S1)

    print("head mean =", y.mean().item())
    print("head std  =", y.std().item())
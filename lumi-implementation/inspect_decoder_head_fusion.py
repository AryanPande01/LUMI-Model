# inspect_decoder_head_fusion.py

from dataset import StockDataset
from model import LUMI

import torch
import numpy as np

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=100
)

x, _ = dataset[0]
x = x.unsqueeze(0)

model = LUMI()

cluster = torch.tensor(
    np.load("cluster_matrix.npy"),
    dtype=torch.float32
)

industry = torch.eye(542)
wiki = torch.eye(542)

with torch.no_grad():

    x = model.input_projection(x)

    semantic = model.semantic_graph(
        industry,
        wiki
    )

    H = model.temporal_encoder(
        x,
        semantic
    )

    H = model.temporal_attention(H)

    H = model.prediction_attention(H)

    print("after pred attn")
    print(H.std().item())

    H = model.decoder(
        H,
        semantic
    )

    print("after decoder")
    print(H.std().item())

    Y = model.head(
        H
    ).squeeze(-1)

    print("after head")
    print(Y.std().item())
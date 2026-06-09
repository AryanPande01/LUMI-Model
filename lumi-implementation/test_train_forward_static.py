# test_train_forward_static.py

import torch
import numpy as np

from model import LUMIStage1
from static_graph_loader import load_static_graphs

device = "cpu"

model = LUMIStage1()

cluster_matrix = torch.tensor(
    np.load("cluster_matrix.npy"),
    dtype=torch.float32
)

industry_graph, wiki_graph = (
    load_static_graphs(device)
)

x = torch.randn(
    2,
    60,
    542,
    1
)

out = model(
    x,
    cluster_matrix,
    industry_graph,
    wiki_graph
)

print(out.shape)
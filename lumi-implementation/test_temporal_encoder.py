import numpy as np
import torch

from attention_layer import GraphAttentionLayer
from dynamic_graph import DynamicGraphBuilder
from temporal_encoder import TemporalSequenceBuilder


cluster_matrix = np.load(
    "cluster_matrix.npy"
)

cluster_matrix = torch.tensor(
    cluster_matrix,
    dtype=torch.float32
)

graph_layer = GraphAttentionLayer(
    in_features=1,
    out_features=8
)

dynamic_graph = DynamicGraphBuilder(
    feature_dim=1,
    hidden_dim=16
)

encoder = TemporalSequenceBuilder(
    graph_layer,
    dynamic_graph
)

dummy_x = torch.randn(
    2,      # batch
    20,     # time
    542,    # stocks
    1       # feature
)

H = encoder(
    dummy_x,
    cluster_matrix
)

print(
    "Output Shape:",
    H.shape
)
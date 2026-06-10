import torch

from attention_layer import GraphAttentionLayer
from temporal_encoder import TemporalSequenceBuilder


graph_layer = GraphAttentionLayer(
    in_features=16,
    out_features=16
)

encoder = TemporalSequenceBuilder(
    graph_layer
)

x = torch.randn(
    2,
    20,
    542,
    16
)

adjacency = torch.randint(
    0,
    2,
    (542, 542)
).float()

out = encoder(
    x,
    adjacency
)

print(out.shape)
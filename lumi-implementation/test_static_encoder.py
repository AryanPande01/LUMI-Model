import torch

from static_graph_encoder import (
    StaticGraphEncoder
)

encoder = StaticGraphEncoder()

x = torch.randn(
    2,
    542
)

adj = torch.ones(
    542,
    542
)

out = encoder(
    x,
    adj
)

print(
    out.shape
)
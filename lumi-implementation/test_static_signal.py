# test_static_signal.py

import torch

from static_graph_encoder import StaticGraphEncoder

encoder = StaticGraphEncoder()

x = torch.randn(
    4,
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

print("Input Mean :", x.mean().item())
print("Input Std  :", x.std().item())

print("Output Mean:", out.mean().item())
print("Output Std :", out.std().item())
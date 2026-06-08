import torch

from temporal_attention import TemporalAttention

H = torch.randn(
    2,
    20,
    542,
    8
)

layer = TemporalAttention()

out = layer(H)

print(
    "Output Shape:",
    out.shape
)
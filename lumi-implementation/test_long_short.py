import torch

from long_short_builder import (
    LongShortBuilder
)

x = torch.randn(
    2,
    20,
    542,
    1
)

builder = LongShortBuilder()

short_seq, long_seq = builder(x)

print(
    "Short:",
    short_seq.shape
)

print(
    "Long:",
    long_seq.shape
)
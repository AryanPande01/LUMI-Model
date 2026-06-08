import torch

from metrics import (
    mae,
    mse,
    information_coefficient,
    rank_ic
)

pred = torch.tensor(
    [1,2,3,4,5],
    dtype=torch.float32
)

target = torch.tensor(
    [1.1,2.2,2.8,4.2,4.9],
    dtype=torch.float32
)

print(
    "MAE:",
    mae(pred,target)
)

print(
    "MSE:",
    mse(pred,target)
)

print(
    "IC:",
    information_coefficient(
        pred,
        target
    )
)

print(
    "Rank IC:",
    rank_ic(
        pred,
        target
    )
)
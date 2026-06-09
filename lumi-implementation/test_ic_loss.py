# test_ic_loss.py

import torch

def ic_loss(pred, target):

    pred = pred.reshape(-1)
    target = target.reshape(-1)

    pred = pred - pred.mean()
    target = target - target.mean()

    numerator = (pred * target).sum()

    denominator = (
        torch.sqrt((pred ** 2).sum())
        * torch.sqrt((target ** 2).sum())
        + 1e-8
    )

    ic = numerator / denominator

    return 1.0 - ic

pred = torch.randn(8, 542)
target = torch.randn(8, 542)

print(ic_loss(pred, target))
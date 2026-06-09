import torch
import torch.nn as nn


pred = torch.randn(8, 542)
target = torch.randn(8, 542)

print(ic_loss(pred, target))
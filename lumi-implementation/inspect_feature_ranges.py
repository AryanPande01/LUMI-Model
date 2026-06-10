from dataset import StockDataset
import numpy as np

d = StockDataset()

x, y = d[0]

print(x.min().item())
print(x.max().item())
print(x.mean().item())
print(x.std().item())
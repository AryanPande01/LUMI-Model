# inspect_targets_after_clip.py

from dataset import StockDataset
import numpy as np

d = StockDataset(
    lookback=100
)

all_y = []

for i in range(len(d)):
    _, y = d[i]
    all_y.append(y.numpy())

all_y = np.concatenate(
    all_y,
    axis=0
)

print("min =", all_y.min())
print("max =", all_y.max())
print("mean =", all_y.mean())
print("std =", all_y.std())

print("95 =", np.percentile(all_y,95))
print("99 =", np.percentile(all_y,99))
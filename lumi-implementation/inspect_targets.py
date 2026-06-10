# inspect_targets.py

from dataset import StockDataset
import numpy as np

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=100
)

all_y = []

for i in range(len(dataset)):
    _, y = dataset[i]
    all_y.append(y.numpy())

all_y = np.concatenate(all_y)

print("target mean =", all_y.mean())
print("positive % =", (all_y > 0).mean())
print("negative % =", (all_y < 0).mean())
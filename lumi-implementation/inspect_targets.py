# inspect_targets.py

from dataset import StockDataset
import numpy as np

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=60
)

ys = []

for i in range(len(dataset)):
    _, y = dataset[i]
    ys.append(y.numpy())

ys = np.concatenate(ys)

print("Target Mean:", ys.mean())
print("Target Std :", ys.std())
print("Target Min :", ys.min())
print("Target Max :", ys.max())
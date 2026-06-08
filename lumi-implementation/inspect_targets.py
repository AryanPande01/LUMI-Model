# inspect_targets.py

from dataset import StockDataset
import numpy as np

dataset = StockDataset(
    "data/LSE/data/price_data.csv",
    "data/LSE/data/gt.csv",
    lookback=60
)

all_y = []

for i in range(len(dataset)):
    _, y = dataset[i]
    all_y.append(y.numpy())

all_y = np.concatenate(all_y)

print("Mean:", all_y.mean())
print("Std :", all_y.std())

print("Zeros:",
      np.mean(all_y == 0))
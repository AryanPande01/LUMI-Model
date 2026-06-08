# inspect_dataset_targets.py

from dataset import StockDataset
import numpy as np

dataset = StockDataset(
    "data/LSE/data/price_data.csv",
    "data/LSE/data/gt.csv",
    lookback=60
)

targets = []

for i in range(len(dataset)):
    _, y = dataset[i]
    targets.append(y.numpy())

targets = np.concatenate(targets)

print("Mean:", targets.mean())
print("Std :", targets.std())
print("Min :", targets.min())
print("Max :", targets.max())
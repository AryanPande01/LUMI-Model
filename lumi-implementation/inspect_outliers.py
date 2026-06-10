from dataset import StockDataset
from data_splitter import create_splits

import numpy as np

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=100
)

_, _, test_set = create_splits(dataset)

values = []

for i in range(len(test_set)):

    _, y = test_set[i]

    values.extend(
        y.numpy().reshape(-1)
    )

values = np.array(values)

print("> 0.2 :", np.sum(np.abs(values) > 0.2))
print("> 0.5 :", np.sum(np.abs(values) > 0.5))
print("> 1.0 :", np.sum(np.abs(values) > 1.0))
print("> 2.0 :", np.sum(np.abs(values) > 2.0))
print("> 5.0 :", np.sum(np.abs(values) > 5.0))

print()

idx = np.argsort(
    np.abs(values)
)[-20:]

print(values[idx])
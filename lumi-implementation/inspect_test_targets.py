from dataset import StockDataset
from data_splitter import create_splits

dataset = StockDataset()

_, _, test_set = create_splits(dataset)

vals = []

for idx in test_set.indices:

    _, y = dataset[idx]

    vals.extend(y.numpy())

import numpy as np

vals = np.array(vals)

print("TEST TARGETS")

print("mean", vals.mean())
print("std", vals.std())
print("min", vals.min())
print("max", vals.max())

print("99%", np.percentile(vals,99))
print("99.9%", np.percentile(vals,99.9))
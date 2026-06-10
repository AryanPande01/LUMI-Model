from dataset import StockDataset
from data_splitter import create_splits

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=100
)

_, _, test_set = create_splits(dataset)

all_targets = []

for i in range(len(test_set)):

    _, y = test_set[i]

    all_targets.append(
        y.numpy()
    )

import numpy as np

all_targets = np.concatenate(
    all_targets,
    axis=0
)

print("shape =", all_targets.shape)

print("min  =", all_targets.min())
print("max  =", all_targets.max())
print("mean =", all_targets.mean())
print("std  =", all_targets.std())

print("95% =", np.percentile(all_targets,95))
print("99% =", np.percentile(all_targets,99))
print("99.9% =", np.percentile(all_targets,99.9))
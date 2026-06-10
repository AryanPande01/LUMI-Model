from dataset import StockDataset
from data_splitter import create_splits

import numpy as np

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=100
)

_, _, test_set = create_splits(dataset)

for i in range(len(test_set)):

    _, y = test_set[i]

    y = y.numpy()

    if np.max(np.abs(y)) > 5:

        print(
            "\nSample:",
            i
        )

        print(
            "max =",
            np.max(y)
        )

        idx = np.unravel_index(
            np.argmax(np.abs(y)),
            y.shape
        )

        print(
            "position =",
            idx
        )

        print(
            "value =",
            y[idx]
        )
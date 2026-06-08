from dataset import StockDataset

from data_splitter import (
    create_splits
)

dataset = StockDataset(
    "data/LSE/data/price_data.csv",
    "data/LSE/data/gt.csv"
)

train_set, val_set, test_set = (
    create_splits(
        dataset
    )
)

print(
    len(train_set)
)

print(
    len(val_set)
)

print(
    len(test_set)
)
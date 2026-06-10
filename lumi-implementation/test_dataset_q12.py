# test_dataset_q12.py

from dataset import StockDataset

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=20
)

x, y = dataset[0]

print()
print("X Shape:", x.shape)
print("Y Shape:", y.shape)
print("Dataset Size:", len(dataset))
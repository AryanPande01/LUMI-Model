# inspect_first_targets.py

from dataset import StockDataset

dataset = StockDataset(
    "data/LSE/data/price_data.csv",
    "data/LSE/data/gt.csv",
    lookback=60
)

for i in range(3):
    _, y = dataset[i]

    print(
        f"Sample {i}",
        y[:10]
    )
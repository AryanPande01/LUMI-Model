from dataset import StockDataset

dataset = StockDataset(
    "data/LSE/data/price_data.csv",
    "data/LSE/data/gt.csv",
    lookback=60
)

x, y = dataset[0]

print(x.shape)
print(y.shape)
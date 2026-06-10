import pandas as pd

price = pd.read_csv("data/LSE/data/price_data.csv")

for i in range(5):
    print(i, price.iloc[i,2:10].values[:5])
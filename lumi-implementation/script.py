import pandas as pd

gt = pd.read_csv("data/LSE/data/gt.csv")

for i in range(10):
    print(i, gt.iloc[i, 2:10].values[:5])
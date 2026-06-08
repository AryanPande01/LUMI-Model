import pandas as pd

gt = pd.read_csv(
    "data/LSE/data/gt.csv"
)

print(gt.iloc[:5, :10])
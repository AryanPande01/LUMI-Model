import pandas as pd

gt = pd.read_csv(
    "data/LSE/data/gt.csv"
)

print(gt.head(15))

print()
print("Rows :", len(gt))
print("Cols :", len(gt.columns))
print()
print(gt.columns[:10])
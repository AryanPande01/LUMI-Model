import pandas as pd

gt = pd.read_csv(
    "data/LSE/data/gt.csv"
)

print(gt.shape)

print("\nRow 0")
print(gt.iloc[0, :10])

print("\nRow 1")
print(gt.iloc[1, :10])

print("\nRow 2")
print(gt.iloc[2, :10])

print("\nRow 3")
print(gt.iloc[3, :10])
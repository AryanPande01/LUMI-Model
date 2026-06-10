import pandas as pd

df = pd.read_csv(
    "data/LSE/data/gt.csv"
)

print(df.tail(20))
print()
print("Rows:", len(df))
import pandas as pd
import numpy as np

gt = pd.read_csv(
    "data/LSE/data/gt.csv"
)

timestamps = gt["Timestamp"]

gt = gt.iloc[1:].reset_index(drop=True)

vals = gt.drop(
    columns=[
        "Unnamed: 0",
        "Timestamp"
    ]
)

arr = vals.values

rows, cols = np.where(arr > 1)

print("Count >", 1, ":", len(rows))

for r, c in zip(rows[:50], cols[:50]):

    print(
        "Row:",
        r,
        "Timestamp:",
        timestamps.iloc[r+1],
        "Stock:",
        c,
        "Value:",
        arr[r, c]
    )
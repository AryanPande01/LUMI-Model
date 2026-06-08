import pandas as pd

gt = pd.read_csv(
    "data/LSE/data/gt.csv"
)

gt = gt.iloc[1:]

gt = gt.drop(
    columns=[
        "Unnamed: 0",
        "Timestamp"
    ]
)

print(
    "Shape:",
    gt.shape
)

print(
    "\nMean:",
    gt.values.mean()
)

print(
    "\nStd:",
    gt.values.std()
)

print(
    "\nMin:",
    gt.values.min()
)

print(
    "\nMax:",
    gt.values.max()
)
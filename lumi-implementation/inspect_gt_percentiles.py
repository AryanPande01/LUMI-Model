import pandas as pd
import numpy as np

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

vals = gt.values.flatten()

print(
    "50%",
    np.percentile(vals,50)
)

print(
    "90%",
    np.percentile(vals,90)
)

print(
    "95%",
    np.percentile(vals,95)
)

print(
    "99%",
    np.percentile(vals,99)
)

print(
    "99.9%",
    np.percentile(vals,99.9)
)

print(
    "MAX",
    vals.max()
)
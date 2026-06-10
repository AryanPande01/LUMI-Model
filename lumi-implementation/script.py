import pandas as pd
import numpy as np

df = pd.read_csv(
    "data/LSE/data/gt.csv"
)

values = (
    df.drop(
        columns=[
            "Unnamed: 0",
            "Timestamp"
        ]
    ).values
)

loc = np.where(values == 8.8953285)

print(loc)
import pandas as pd

df = pd.read_csv(
    "data/LSE/data/gt.csv"
)

print(
    df.iloc[1128:1138, [0,1,501+2]]
)
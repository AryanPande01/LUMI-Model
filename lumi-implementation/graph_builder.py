import pandas as pd
import numpy as np

def build_adjacency(csv_path):

    df = pd.read_csv(csv_path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    num_nodes = (
        max(
            df["Node"].max(),
            df["Cor_Node"].max()
        ) + 1
    )

    adj = np.zeros(
        (num_nodes, num_nodes),
        dtype=np.float32
    )

    for _, row in df.iterrows():

        i = int(row["Node"])
        j = int(row["Cor_Node"])
        val = float(row["Cor"])

        adj[i][j] = val

    return adj
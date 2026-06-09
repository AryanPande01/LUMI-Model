import torch
import numpy as np

from dataset import StockDataset
from model import LUMIStage1
from static_graph_loader import load_static_graphs

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=60
)

cluster_matrix = np.load(
    "cluster_matrix.npy"
)

cluster_matrix = torch.tensor(
    cluster_matrix,
    dtype=torch.float32
)

industry_graph, wiki_graph = load_static_graphs(
    "cpu"
)

model = LUMIStage1()

model.load_state_dict(
    torch.load(
        "final_model.pth",
        map_location="cpu"
    )
)

model.eval()

preds = []

with torch.no_grad():

    for i in range(50):

        x, _ = dataset[i]

        p = model(
            x.unsqueeze(0),
            cluster_matrix,
            industry_graph,
            wiki_graph
        )

        preds.append(
            p.flatten().numpy()
        )

preds = np.concatenate(preds)

print("Mean:", preds.mean())
print("Std :", preds.std())
print("Min :", preds.min())
print("Max :", preds.max())
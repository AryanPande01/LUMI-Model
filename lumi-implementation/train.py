from dataset import StockDataset
from graph_builder import build_adjacency
from model import LUMIStage1

import torch

# Dataset
dataset = StockDataset(
    "data/LSE/data/price_data.csv",
    "data/LSE/data/gt.csv",
    lookback=20
)

x, y = dataset[0]

print("Input Shape :", x.shape)
print("Target Shape:", y.shape)

# Graphs
industry_graph = build_adjacency(
    "data/LSE/graph_data/industry_adjacency.csv"
)

wiki_graph = build_adjacency(
    "data/LSE/graph_data/wiki_adjacency.csv"
)

semantic_graph = industry_graph + wiki_graph

semantic_graph = torch.tensor(
    semantic_graph,
    dtype=torch.float32
)

# Model
model = LUMIStage1(
    num_nodes=542
)

# Add batch dimension
batch_x = x.unsqueeze(0)

pred = model(
    batch_x,
    semantic_graph
)

print("Prediction Shape:", pred.shape)
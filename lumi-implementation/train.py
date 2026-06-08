from dataset import StockDataset
from graph_builder import build_adjacency
from model import LUMIStage1

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

# ------------------------
# Dataset
# ------------------------

dataset = StockDataset(
    "data/LSE/data/price_data.csv",
    "data/LSE/data/gt.csv",
    lookback=20
)

print("Dataset Size:", len(dataset))

# ------------------------
# Dataloader
# ------------------------

train_loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True
)

# ------------------------
# Graphs
# ------------------------

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

# ------------------------
# Model
# ------------------------

model = LUMIStage1(
    num_nodes=542
)

criterion = nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)

# ------------------------
# Training
# ------------------------

epochs = 5

for epoch in range(epochs):

    model.train()

    epoch_loss = 0

    for x, y in train_loader:

        optimizer.zero_grad()

        pred = model(
            x,
            semantic_graph
        )

        loss = criterion(
            pred,
            y
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(train_loader)

    print(
        f"Epoch {epoch+1}/{epochs} | Loss = {avg_loss:.6f}"
    )
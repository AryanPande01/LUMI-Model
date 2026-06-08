from dataset import StockDataset
from model import LUMIStage1

import numpy as np
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
# Cluster Matrix
# ------------------------

cluster_matrix = np.load(
    "cluster_matrix.npy"
)

cluster_matrix = torch.tensor(
    cluster_matrix,
    dtype=torch.float32
)

print(
    "Cluster Matrix Shape:",
    cluster_matrix.shape
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
            cluster_matrix
        )

        loss = criterion(
            pred,
            y
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = (
        epoch_loss /
        len(train_loader)
    )

    print(
        f"Epoch {epoch+1}/{epochs} | Loss = {avg_loss:.6f}"
    )
from dataset import StockDataset
from model import LUMIStage1

from data_splitter import create_splits
from evaluate import evaluate_model

from metrics import (
    mae,
    mse,
    information_coefficient,
    rank_ic
)

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm


# ------------------------
# Device
# ------------------------

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Using Device:",
    device
)


# ------------------------
# Dataset
# ------------------------

dataset = StockDataset(
    "data/LSE/data/price_data.csv",
    "data/LSE/data/gt.csv",
    lookback=60
)

print(
    "Dataset Size:",
    len(dataset)
)


# ------------------------
# Dataset Split
# ------------------------

train_set, val_set, test_set = (
    create_splits(
        dataset
    )
)

print(
    "Train Size:",
    len(train_set)
)

print(
    "Val Size:",
    len(val_set)
)

print(
    "Test Size:",
    len(test_set)
)


# ------------------------
# Dataloaders
# ------------------------

train_loader = DataLoader(
    train_set,
    batch_size=8,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=8,
    shuffle=False
)

test_loader = DataLoader(
    test_set,
    batch_size=8,
    shuffle=False
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
).to(device)

print(
    "Cluster Matrix Shape:",
    cluster_matrix.shape
)


# ------------------------
# Model
# ------------------------

model = LUMIStage1(
    num_nodes=542
).to(device)

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ------------------------
# Training
# ------------------------

best_val_loss = float("inf")

epochs = 20

for epoch in range(epochs):

    model.train()

    epoch_loss = 0
    epoch_mae = 0
    epoch_mse = 0
    epoch_ic = 0
    epoch_rank_ic = 0

    batches = 0

    for x, y in tqdm(
        train_loader,
        desc=f"Epoch {epoch+1}"
    ):

        x = x.to(device)
        y = y.to(device)

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

        epoch_mae += mae(
            pred.detach(),
            y.detach()
        )

        epoch_mse += mse(
            pred.detach(),
            y.detach()
        )

        epoch_ic += (
            information_coefficient(
                pred.detach(),
                y.detach()
            )
        )

        epoch_rank_ic += (
            rank_ic(
                pred.detach(),
                y.detach()
            )
        )

        batches += 1

    # ------------------------
    # Train Metrics
    # ------------------------

    print()

    print(
        f"Epoch {epoch+1}/{epochs}"
    )

    print(
        "\nTRAIN"
    )

    print(
        f"Loss     : {epoch_loss/batches:.6f}"
    )

    print(
        f"MAE      : {epoch_mae/batches:.6f}"
    )

    print(
        f"MSE      : {epoch_mse/batches:.6f}"
    )

    print(
        f"IC       : {epoch_ic/batches:.6f}"
    )

    print(
        f"Rank IC  : {epoch_rank_ic/batches:.6f}"
    )

    # ------------------------
    # Validation Metrics
    # ------------------------

    val_metrics = evaluate_model(
        model,
        val_loader,
        cluster_matrix,
        criterion,
        device
    )

    print()

    print(
        "VALIDATION"
    )

    print(
        f"Loss     : {val_metrics['loss']:.6f}"
    )

    print(
        f"MAE      : {val_metrics['mae']:.6f}"
    )

    print(
        f"MSE      : {val_metrics['mse']:.6f}"
    )

    print(
        f"IC       : {val_metrics['ic']:.6f}"
    )

    print(
        f"Rank IC  : {val_metrics['rank_ic']:.6f}"
    )

    # ------------------------
    # Save Best Model
    # ------------------------

    if val_metrics["loss"] < best_val_loss:

        best_val_loss = val_metrics["loss"]

        torch.save(
            model.state_dict(),
            "best_model.pth"
        )

        print(
            "Saved Best Model"
        )


# ------------------------
# Final Test Evaluation
# ------------------------

print()

print(
    "FINAL TEST EVALUATION"
)

test_metrics = evaluate_model(
    model,
    test_loader,
    cluster_matrix,
    criterion,
    device
)

print(
    f"Loss     : {test_metrics['loss']:.6f}"
)

print(
    f"MAE      : {test_metrics['mae']:.6f}"
)

print(
    f"MSE      : {test_metrics['mse']:.6f}"
)

print(
    f"IC       : {test_metrics['ic']:.6f}"
)

print(
    f"Rank IC  : {test_metrics['rank_ic']:.6f}"
)

# ------------------------
# Save Final Model
# ------------------------

torch.save(
    model.state_dict(),
    "final_model.pth"
)

print(
    "\nFinal model saved as final_model.pth"
)
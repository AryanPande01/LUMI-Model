from dataset import StockDataset
from model import LUMI

from data_splitter import create_splits
from evaluate import evaluate_model

from static_graph_loader import load_static_graphs

from metrics import (
    mae,
    mse,
    rmse,
    mape,
    directional_accuracy,
    information_coefficient,
    rank_ic
)

import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from tqdm import tqdm

torch.cuda.empty_cache()

def ic_loss(pred, target):

    horizon_losses = []

    for q in range(pred.shape[1]):

        p = pred[:, q, :].reshape(-1)
        t = target[:, q, :].reshape(-1)

        p = p - p.mean()
        t = t - t.mean()

        numerator = (p * t).sum()

        denominator = (
            torch.sqrt((p ** 2).sum())
            *
            torch.sqrt((t ** 2).sum())
            +
            1e-8
        )

        ic = numerator / denominator

        horizon_losses.append(
            1.0 - ic
        )

    return torch.stack(
        horizon_losses
    ).mean()

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
    data_dir="data/LSE/data",
    lookback=20
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
    batch_size=2,
    shuffle=True
)

val_loader = DataLoader(
    val_set,
    batch_size=2,
    shuffle=False
)

test_loader = DataLoader(
    test_set,
    batch_size=2,
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

industry_graph, wiki_graph = (
    load_static_graphs(device)
)

print(
    "Industry Graph Shape:",
    industry_graph.shape
)

print(
    "Wiki Graph Shape:",
    wiki_graph.shape
)

print(
    "Industry Non-Zero:",
    (industry_graph != 0).sum().item()
)

print(
    "Wiki Non-Zero:",
    (wiki_graph != 0).sum().item()
)
# ------------------------
# Model
# ------------------------

model = LUMI(
    num_nodes=542,
    hidden_dim=16,
    horizon=12
).to(device)

criterion = nn.L1Loss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001
)


# ------------------------
# Training
# -----------------------

best_val_loss = float("inf")

epochs = 5

for epoch in range(epochs):

    model.train()

    epoch_loss = 0
    epoch_mae = 0
    epoch_mse = 0
    epoch_rmse = 0
    epoch_mape = 0
    epoch_da = 0
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
            cluster_matrix,
            industry_graph,
            wiki_graph
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

        epoch_rmse += rmse(
            pred.detach(),
            y.detach()
        )

        epoch_mape += mape(
            pred.detach(),
            y.detach()
        )

        epoch_da += directional_accuracy(
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
        f"RMSE     : {epoch_rmse/batches:.6f}"
    )

    print(
        f"MAPE     : {epoch_mape/batches:.6f}"
    )

    print(
        f"DA (%)   : {epoch_da/batches:.2f}"
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
        industry_graph,
        wiki_graph,
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
        f"RMSE     : {val_metrics['rmse']:.6f}"
    )

    print(
        f"MAPE     : {val_metrics['mape']:.6f}"
    )

    print(
        f"DA (%)   : {val_metrics['da']:.2f}"
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
# Load Best Model
# ------------------------

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

print(
    "\nLoaded Best Model"
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
    industry_graph,
    wiki_graph,
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
    f"RMSE     : {test_metrics['rmse']:.6f}"
)

print(
    f"MAPE     : {test_metrics['mape']:.6f}"
)

print(
    f"DA (%)   : {test_metrics['da']:.2f}"
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
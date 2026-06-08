import torch

from metrics import (
    mae,
    mse,
    information_coefficient,
    rank_ic
)


def evaluate_model(
    model,
    loader,
    cluster_matrix,
    criterion,
    device
):

    model.eval()

    total_loss = 0
    total_mae = 0
    total_mse = 0
    total_ic = 0
    total_rank_ic = 0

    batches = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            pred = model(
                x,
                cluster_matrix
            )

            loss = criterion(
                pred,
                y
            )

            total_loss += (
                loss.item()
            )

            total_mae += mae(
                pred,
                y
            )

            total_mse += mse(
                pred,
                y
            )

            total_ic += (
                information_coefficient(
                    pred,
                    y
                )
            )

            total_rank_ic += (
                rank_ic(
                    pred,
                    y
                )
            )

            batches += 1

    return {

        "loss":
            total_loss / batches,

        "mae":
            total_mae / batches,

        "mse":
            total_mse / batches,

        "ic":
            total_ic / batches,

        "rank_ic":
            total_rank_ic / batches
    }
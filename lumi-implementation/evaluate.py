import torch

from metrics import (
    mae,
    mse,
    rmse,
    mape,
    directional_accuracy,
    information_coefficient,
    rank_ic
)


def evaluate_model(
    model,
    loader,
    cluster_matrix,
    industry_graph,
    wiki_graph,
    criterion,
    device
):

    model.eval()

    total_loss = 0

    total_mae = 0
    total_mse = 0
    total_rmse = 0
    total_mape = 0
    total_da = 0

    total_ic = 0
    total_rank_ic = 0

    batches = 0

    with torch.no_grad():

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

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

            total_rmse += rmse(
                pred,
                y
            )

            total_mape += mape(
                pred,
                y
            )

            total_da += directional_accuracy(
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

        "rmse":
            total_rmse / batches,

        "mape":
            total_mape / batches,

        "da":
            total_da / batches,

        "ic":
            total_ic / batches,

        "rank_ic":
            total_rank_ic / batches
    }
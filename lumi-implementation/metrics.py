import torch


def mae(pred, target):

    return torch.mean(
        torch.abs(
            pred - target
        )
    ).item()


def mse(pred, target):

    return torch.mean(
        (
            pred - target
        ) ** 2
    ).item()


def information_coefficient(
    pred,
    target
):

    pred = pred.flatten()

    target = target.flatten()

    pred = pred - pred.mean()

    target = (
        target - target.mean()
    )

    numerator = (
        pred * target
    ).sum()

    denominator = torch.sqrt(
        (
            pred ** 2
        ).sum()
    ) * torch.sqrt(
        (
            target ** 2
        ).sum()
    )

    if denominator == 0:
        return 0.0

    return (
        numerator /
        denominator
    ).item()


def rank_ic(
    pred,
    target
):

    pred_rank = torch.argsort(
        torch.argsort(pred.flatten())
    ).float()

    target_rank = torch.argsort(
        torch.argsort(target.flatten())
    ).float()

    return information_coefficient(
        pred_rank,
        target_rank
    )
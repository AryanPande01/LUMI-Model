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

    if pred.dim() == 3:

        ics = []

        for q in range(pred.shape[1]):

            p = pred[:, q, :].reshape(-1)
            t = target[:, q, :].reshape(-1)

            p = p - p.mean()
            t = t - t.mean()

            denom = (
                torch.sqrt((p ** 2).sum())
                *
                torch.sqrt((t ** 2).sum())
            )

            if denom > 0:

                ics.append(
                    (
                        (p * t).sum()
                        /
                        denom
                    ).item()
                )

        return sum(ics) / len(ics)

    p = pred.reshape(-1)
    t = target.reshape(-1)

    p = p - p.mean()
    t = t - t.mean()

    denom = (
        torch.sqrt((p ** 2).sum())
        *
        torch.sqrt((t ** 2).sum())
    )

    if denom == 0:
        return 0.0

    return (
        (p * t).sum()
        /
        denom
    ).item()


def rank_ic(
    pred,
    target
):

    if pred.dim() == 3:

        scores = []

        for q in range(pred.shape[1]):

            p = pred[:, q, :]
            t = target[:, q, :]

            p_rank = torch.argsort(
                torch.argsort(
                    p.reshape(-1)
                )
            ).float()

            t_rank = torch.argsort(
                torch.argsort(
                    t.reshape(-1)
                )
            ).float()

            scores.append(
                information_coefficient(
                    p_rank,
                    t_rank
                )
            )

        return sum(scores) / len(scores)

    p_rank = torch.argsort(
        torch.argsort(
            pred.reshape(-1)
        )
    ).float()

    t_rank = torch.argsort(
        torch.argsort(
            target.reshape(-1)
        )
    ).float()

    return information_coefficient(
        p_rank,
        t_rank
    )
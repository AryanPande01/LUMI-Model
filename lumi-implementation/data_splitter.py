from torch.utils.data import Subset


def create_splits(dataset):

    total = len(dataset)

    train_end = int(
        total * 0.70
    )

    val_end = int(
        total * 0.85
    )

    train_indices = list(
        range(0, train_end)
    )

    val_indices = list(
        range(train_end, val_end)
    )

    test_indices = list(
        range(val_end, total)
    )

    train_set = Subset(
        dataset,
        train_indices
    )

    val_set = Subset(
        dataset,
        val_indices
    )

    test_set = Subset(
        dataset,
        test_indices
    )

    return (
        train_set,
        val_set,
        test_set
    )
from torch.utils.data import (
    random_split
)


def create_splits(
    dataset
):

    total = len(dataset)

    train_size = int(
        total * 0.70
    )

    val_size = int(
        total * 0.15
    )

    test_size = (
        total
        - train_size
        - val_size
    )

    train_set, val_set, test_set = (
        random_split(
            dataset,
            [
                train_size,
                val_size,
                test_size
            ]
        )
    )

    return (
        train_set,
        val_set,
        test_set
    )
    
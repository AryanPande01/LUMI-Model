import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):

    def __init__(
        self,
        price_csv,
        gt_csv,
        lookback=60
    ):

        # -----------------------
        # Load price data
        # -----------------------

        price_df = pd.read_csv(
            price_csv
        )

        # remove fake row
        price_df = (
            price_df.iloc[1:]
            .reset_index(drop=True)
        )

        self.timestamps = (
            price_df["Timestamp"]
        )

        self.price_data = (
            price_df.drop(
                columns=[
                    "Unnamed: 0",
                    "Timestamp"
                ]
            )
            .values
            .astype(np.float32)
        )

        # -----------------------
        # Load GT data
        # -----------------------

        gt_df = pd.read_csv(
            gt_csv
        )

        # remove:
        # row 0 -> fake row
        # row 1 -> price row
        gt_df = (
            gt_df.iloc[2:]
            .reset_index(drop=True)
        )

        self.gt_data = (
            gt_df.drop(
                columns=[
                    "Unnamed: 0",
                    "Timestamp"
                ]
            )
            .values
            .astype(np.float32)
        )

        self.lookback = lookback

        # -----------------------
        # Align lengths
        # -----------------------

        min_len = min(
            len(self.price_data),
            len(self.gt_data)
        )

        self.price_data = (
            self.price_data[:min_len]
        )

        self.gt_data = (
            self.gt_data[:min_len]
        )

    def __len__(self):

        return (
            len(self.price_data)
            - self.lookback
        )

    def __getitem__(
        self,
        idx
    ):

        x = self.price_data[
            idx :
            idx + self.lookback
        ]

        y = self.gt_data[
            idx + self.lookback
        ]

        x = torch.tensor(
            x,
            dtype=torch.float32
        ).unsqueeze(-1)

        y = torch.tensor(
            y,
            dtype=torch.float32
        )

        return x, y
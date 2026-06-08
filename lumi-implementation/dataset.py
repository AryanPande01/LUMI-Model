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
        price_df = pd.read_csv(price_csv)

        # Remove fake first row
        price_df = price_df.iloc[1:].reset_index(drop=True)

        self.timestamps = price_df["Timestamp"]

        self.price_data = price_df.drop(
            columns=["Unnamed: 0", "Timestamp"]
        ).values.astype(np.float32)

        # -----------------------
        # Load ground truth data
        # -----------------------
        gt_df = pd.read_csv(gt_csv)

        # Remove fake first row
        gt_df = gt_df.iloc[1:].reset_index(drop=True)

        self.gt_data = gt_df.drop(
            columns=["Unnamed: 0", "Timestamp"]
        ).values.astype(np.float32)

        self.lookback = lookback

    def __len__(self):
        return len(self.price_data) - self.lookback

    def __getitem__(self, idx):

        # Past 20 days prices
        x = self.price_data[
            idx : idx + self.lookback
        ]

        # Next-day return target
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
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset


class StockDataset(Dataset):

    def __init__(
        self,
        data_dir="data/LSE/data",
        lookback=60
    ):

        self.lookback = lookback
        self.horizon = 12

        # -----------------------
        # Price
        # -----------------------

        price_df = pd.read_csv(
            f"{data_dir}/price_data.csv"
        )

        price_df = (
            price_df.iloc[1:]
            .reset_index(drop=True)
        )

        price_values = (
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
        # MA5
        # -----------------------

        ma5_df = pd.read_csv(
            f"{data_dir}/data_ma_5.csv"
        )

        ma5_df = (
            ma5_df.iloc[1:]
            .reset_index(drop=True)
        )

        ma5_values = (
            ma5_df.drop(
                columns=[
                    "Unnamed: 0",
                    "Timestamp"
                ]
            )
            .values
            .astype(np.float32)
        )

        # -----------------------
        # MA10
        # -----------------------

        ma10_df = pd.read_csv(
            f"{data_dir}/data_ma_10.csv"
        )

        ma10_df = (
            ma10_df.iloc[1:]
            .reset_index(drop=True)
        )

        ma10_values = (
            ma10_df.drop(
                columns=[
                    "Unnamed: 0",
                    "Timestamp"
                ]
            )
            .values
            .astype(np.float32)
        )

        # -----------------------
        # MA20
        # -----------------------

        ma20_df = pd.read_csv(
            f"{data_dir}/data_ma_20.csv"
        )

        ma20_df = (
            ma20_df.iloc[1:]
            .reset_index(drop=True)
        )

        ma20_values = (
            ma20_df.drop(
                columns=[
                    "Unnamed: 0",
                    "Timestamp"
                ]
            )
            .values
            .astype(np.float32)
        )

        # -----------------------
        # MA30
        # -----------------------

        ma30_df = pd.read_csv(
            f"{data_dir}/data_ma_30.csv"
        )

        ma30_df = (
            ma30_df.iloc[1:]
            .reset_index(drop=True)
        )

        ma30_values = (
            ma30_df.drop(
                columns=[
                    "Unnamed: 0",
                    "Timestamp"
                ]
            )
            .values
            .astype(np.float32)
        )

        # -----------------------
        # GT
        # -----------------------

        gt_df = pd.read_csv(
            f"{data_dir}/gt.csv"
        )

        gt_df = (
            gt_df.iloc[2:]
            .reset_index(drop=True)
        )

        gt_values = (
            gt_df.drop(
                columns=[
                    "Unnamed: 0",
                    "Timestamp"
                ]
            )
            .values
            .astype(np.float32)
        )

        gt_values = np.clip(
            gt_values,
            -0.2,
            0.2
        )


        # -----------------------
        # Align
        # -----------------------

        min_len = min(
            len(price_values),
            len(ma5_values),
            len(ma10_values),
            len(ma20_values),
            len(ma30_values),
            len(gt_values)
        )

        price_values = price_values[:min_len]
        ma5_values = ma5_values[:min_len]
        ma10_values = ma10_values[:min_len]
        ma20_values = ma20_values[:min_len]
        ma30_values = ma30_values[:min_len]
        gt_values = gt_values[:min_len]

        # -----------------------
        # Feature Tensor
        # -----------------------

        self.features = np.stack(
            [
                price_values,
                ma5_values,
                ma10_values,
                ma20_values,
                ma30_values
            ],
            axis=-1
        )

        feature_mean = self.features.mean(
            axis=(0, 1),
            keepdims=True
        )

        feature_std = self.features.std(
            axis=(0, 1),
            keepdims=True
        )

        self.features = (
            self.features - feature_mean
        ) / (
            feature_std + 1e-8
        )

        self.targets = gt_values

        print(
            "Feature Tensor Shape:",
            self.features.shape
        )

    def __len__(self):

        return (
            len(self.features)
            - self.lookback
            - self.horizon
            + 1
        )

    def __getitem__(
        self,
        idx
    ):

        # -----------------------
        # Input Sequence
        # -----------------------

        x = self.features[
            idx:
            idx + self.lookback
        ]

        # -----------------------
        # Future 12-Day Targets
        # -----------------------

        y = self.targets[
            idx + self.lookback:
            idx + self.lookback + self.horizon
        ]

        x = torch.tensor(
            x,
            dtype=torch.float32
        )

        y = torch.tensor(
            y,
            dtype=torch.float32
        )

        return x, y
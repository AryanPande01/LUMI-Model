import torch
import torch.nn as nn


class PredictionAttention(nn.Module):

    def __init__(
        self,
        feature_dim=16
    ):
        super().__init__()

        self.query = nn.Linear(
            feature_dim,
            feature_dim
        )

        self.key = nn.Linear(
            feature_dim,
            feature_dim
        )

        self.value = nn.Linear(
            feature_dim,
            feature_dim
        )

    def forward(self, H):

        # H
        # [B,N,D]

        Q = self.query(H)

        K = self.key(H)

        V = self.value(H)

        scores = torch.matmul(
            Q,
            K.transpose(-1, -2)
        )

        scores = scores / (
            H.shape[-1] ** 0.5
        )

        alpha = torch.softmax(
            scores,
            dim=-1
        )

        out = torch.matmul(
            alpha,
            V
        )

        return out
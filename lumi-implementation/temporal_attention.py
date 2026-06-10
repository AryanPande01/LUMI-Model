import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):

    def __init__(
        self,
        feature_dim=8
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
        # [B,T,N,D]

        B, T, N, D = H.shape

        H = H.permute(
            0,
            2,
            1,
            3
        )

        H = H.reshape(
            B * N,
            T,
            D
        )

        Q = self.query(H)

        K = self.key(H)

        V = self.value(H)

        scores = torch.matmul(
            Q,
            K.transpose(-1, -2)
        )

        scores = scores / (
            D ** 0.5
        )

        alpha = torch.softmax(
            scores,
            dim=-1
        )

        out = torch.matmul(
            alpha,
            V
        )

        # ----------------------
        # IMPORTANT CHANGE
        # ----------------------

        out = out[:, -1, :]

        out = out.reshape(
            B,
            N,
            D
        )

        return out
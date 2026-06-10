import torch
import torch.nn as nn


class PredictionAttention(nn.Module):

    def __init__(
        self,
        feature_dim=16,
        horizon=12
    ):
        super().__init__()

        self.horizon = horizon

        self.query = nn.Parameter(
            torch.randn(
                horizon,
                feature_dim
            )
        )

        self.key = nn.Linear(
            feature_dim,
            feature_dim
        )

        self.value = nn.Linear(
            feature_dim,
            feature_dim
        )

    def forward(
        self,
        H
    ):

        # H
        # [B,N,D]

        B, N, D = H.shape

        K = self.key(
            H
        )

        V = self.value(
            H
        )

        horizon_outputs = []

        for q in range(
            self.horizon
        ):

            q_vec = self.query[q]

            q_vec = q_vec.view(
                1,
                1,
                D
            )

            scores = (
                K * q_vec
            ).sum(
                dim=-1,
                keepdim=True
            )

            scores = scores / (
                D ** 0.5
            )

            alpha = torch.softmax(
                scores,
                dim=1
            )

            future_embedding = V

            horizon_outputs.append(
                future_embedding
            )

        out = torch.stack(
            horizon_outputs,
            dim=1
        )

        # [B,Q,N,D]

        return out
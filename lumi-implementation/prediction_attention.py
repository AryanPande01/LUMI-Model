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
        # [B,T,N,D]

        B, T, N, D = H.shape

        K = self.key(H)

        V = self.value(H)

        outputs = []

        for q in range(self.horizon):

            q_vec = self.query[q]

            q_vec = q_vec.view(
                1,
                1,
                1,
                D
            )

            scores = (
                K * q_vec
            ).sum(
                dim=-1
            )

            scores = scores / (
                D ** 0.5
            )

            # attention over TIME

            alpha = torch.softmax(
                scores,
                dim=1
            )

            alpha = alpha.unsqueeze(
                -1
            )

            future_embedding = (
                alpha * V
            ).sum(
                dim=1
            )

            outputs.append(
                future_embedding
            )

        out = torch.stack(
            outputs,
            dim=1
        )

        # [B,Q,N,D]

        return out
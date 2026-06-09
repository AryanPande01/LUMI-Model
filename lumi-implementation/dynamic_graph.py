import torch
import torch.nn as nn


class DynamicGraphBuilder(nn.Module):

    def __init__(
        self,
        feature_dim=1,
        hidden_dim=32
    ):
        super().__init__()

        self.query = nn.Linear(
            feature_dim,
            hidden_dim,
            bias=False
        )

        self.key = nn.Linear(
            feature_dim,
            hidden_dim,
            bias=False
        )

    def forward(
        self,
        node_features,
        cluster_matrix
    ):

        # [B,N,5]

        Q = self.query(
            node_features
        )

        K = self.key(
            node_features
        )

        scores = torch.matmul(
            Q,
            K.transpose(-1, -2)
        )

        scores = scores / (
            Q.shape[-1] ** 0.5
        )

        mask = (
            cluster_matrix == 0
        )

        scores = scores.masked_fill(
            mask,
            -1e9
        )

        attention = torch.softmax(
            scores,
            dim=-1
        )

        return attention
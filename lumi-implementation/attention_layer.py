import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):

    def __init__(
        self,
        in_features=5,
        out_features=16
    ):
        super().__init__()

        self.query = nn.Linear(
            in_features,
            out_features,
            bias=False
        )

        self.key = nn.Linear(
            in_features,
            out_features,
            bias=False
        )

        self.value = nn.Linear(
            in_features,
            out_features,
            bias=False
        )

    def forward(
        self,
        node_features,
        adjacency
    ):

        # node_features
        # [B,N,F]

        Q = self.query(
            node_features
        )

        K = self.key(
            node_features
        )

        V = self.value(
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
            adjacency == 0
        )

        scores = scores.masked_fill(
            mask,
            -1e9
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
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):

    def __init__(
        self,
        in_features,
        out_features
    ):
        super().__init__()

        self.W = nn.Linear(
            in_features,
            out_features,
            bias=False
        )

        self.attn = nn.Linear(
            2 * out_features,
            1,
            bias=False
        )

    def forward(
        self,
        node_features,
        adjacency
    ):

        # node_features:
        # [B,N,1]

        h = self.W(
            node_features
        )

        B, N, Fdim = h.shape

        h_i = h.unsqueeze(2).repeat(
            1, 1, N, 1
        )

        h_j = h.unsqueeze(1).repeat(
            1, N, 1, 1
        )

        concat = torch.cat(
            [h_i, h_j],
            dim=-1
        )

        e = F.leaky_relu(
            self.attn(concat)
        ).squeeze(-1)

        mask = adjacency == 0

        e = e.masked_fill(
            mask,
            -1e9
        )

        alpha = torch.softmax(
            e,
            dim=-1
        )

        out = torch.matmul(
            alpha,
            h
        )

        return out
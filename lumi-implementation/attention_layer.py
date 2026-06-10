import torch
import torch.nn as nn


class GraphAttentionLayer(nn.Module):

    def __init__(
        self,
        in_features=16,
        out_features=16,
        num_heads=4
    ):
        super().__init__()

        self.num_heads = num_heads

        self.head_dim = (
            out_features // num_heads
        )

        self.query = nn.ModuleList([
            nn.Linear(
                in_features,
                self.head_dim,
                bias=False
            )
            for _ in range(num_heads)
        ])

        self.key = nn.ModuleList([
            nn.Linear(
                in_features,
                self.head_dim,
                bias=False
            )
            for _ in range(num_heads)
        ])

        self.value = nn.ModuleList([
            nn.Linear(
                in_features,
                self.head_dim,
                bias=False
            )
            for _ in range(num_heads)
        ])

        self.output_projection = nn.Linear(
            out_features,
            out_features
        )

    def forward(
        self,
        node_features,
        adjacency
    ):

        head_outputs = []

        for h in range(
            self.num_heads
        ):

            Q = self.query[h](
                node_features
            )

            K = self.key[h](
                node_features
            )

            V = self.value[h](
                node_features
            )

            scores = torch.matmul(
                Q,
                K.transpose(-1, -2)
            )

            scores = scores / (
                self.head_dim ** 0.5
            )

            scores = scores.masked_fill(
                adjacency == 0,
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

            head_outputs.append(
                out
            )

        out = torch.cat(
            head_outputs,
            dim=-1
        )

        out = self.output_projection(
            out
        )

        return out
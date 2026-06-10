import torch
import torch.nn as nn


class TemporalAttention(nn.Module):

    def __init__(
        self,
        feature_dim=16,
        num_heads=8
    ):
        super().__init__()

        self.num_heads = num_heads

        self.head_dim = (
            feature_dim // num_heads
        )

        self.query = nn.ModuleList([
            nn.Linear(
                feature_dim,
                self.head_dim,
                bias=False
            )
            for _ in range(num_heads)
        ])

        self.key = nn.ModuleList([
            nn.Linear(
                feature_dim,
                self.head_dim,
                bias=False
            )
            for _ in range(num_heads)
        ])

        self.value = nn.ModuleList([
            nn.Linear(
                feature_dim,
                self.head_dim,
                bias=False
            )
            for _ in range(num_heads)
        ])

        self.output_projection = nn.Linear(
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

        H = H.permute(
            0,
            2,
            1,
            3
        )

        # [B,N,T,D]

        H = H.reshape(
            B * N,
            T,
            D
        )

        # [B*N,T,D]

        head_outputs = []

        for h in range(
            self.num_heads
        ):

            Q = self.query[h](
                H
            )

            K = self.key[h](
                H
            )

            V = self.value[h](
                H
            )

            scores = torch.matmul(
                Q,
                K.transpose(-1, -2)
            )

            scores = scores / (
                self.head_dim ** 0.5
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

        out = out.reshape(
            B,
            N,
            T,
            D
        )

        out = out.permute(
            0,
            2,
            1,
            3
        )

        # [B,T,N,D]

        return out
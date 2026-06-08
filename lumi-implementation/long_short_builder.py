import torch
import torch.nn as nn


class LongShortBuilder(nn.Module):

    def __init__(
        self,
        short_window=20,
        long_gap=5,
        long_length=20
    ):
        super().__init__()

        self.short_window = short_window
        self.long_gap = long_gap
        self.long_length = long_length

    def forward(self, x):

        # x
        # [B,T,N,1]

        short_seq = x

        B, T, N, C = x.shape

        indices = []

        current = T - 1

        for _ in range(
            self.long_length
        ):
            indices.append(
                max(current, 0)
            )

            current -= (
                self.long_gap
            )

        indices.reverse()

        long_seq = x[
            :,
            indices,
            :,
            :
        ]

        return (
            short_seq,
            long_seq
        )
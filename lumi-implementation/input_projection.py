import torch
import torch.nn as nn


class InputProjection(nn.Module):

    def __init__(
        self,
        input_dim=5,
        hidden_dim=16
    ):
        super().__init__()

        self.fc = nn.Linear(
            input_dim,
            hidden_dim
        )

    def forward(
        self,
        x
    ):

        # x
        # [B,T,N,C]

        H = self.fc(
            x
        )

        # [B,T,N,D]

        return H
import torch
import torch.nn as nn
import torch.nn.functional as F


class HorizonFusion(nn.Module):

    def __init__(
        self,
        horizon=12
    ):
        super().__init__()

        self.horizon = horizon

        self.weights = nn.Parameter(
            torch.randn(
                horizon,
                4
            )
        )

    def forward(
        self,
        y1,
        y2,
        y3,
        y4
    ):

        # all inputs
        # [B,Q,N]

        B, Q, N = y1.shape

        outputs = []

        for q in range(Q):

            alpha = F.softmax(
                self.weights[q],
                dim=0
            )

            fused = (
                alpha[0] * y1[:, q, :]
                +
                alpha[1] * y2[:, q, :]
                +
                alpha[2] * y3[:, q, :]
                +
                alpha[3] * y4[:, q, :]
            )

            outputs.append(
                fused
            )

        out = torch.stack(
            outputs,
            dim=1
        )

        # [B,Q,N]

        return out
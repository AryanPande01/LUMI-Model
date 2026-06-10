import torch
import torch.nn as nn


class TemporalSequenceBuilder(nn.Module):

    def __init__(
        self,
        graph_layer
    ):
        super().__init__()

        self.graph_layer = graph_layer

    def forward(
        self,
        x,
        adjacency
    ):

        # x
        # [B,T,N,F]

        outputs = []

        T = x.shape[1]

        for t in range(T):

            xt = x[:, t, :, :]

            Ht = self.graph_layer(
                xt,
                adjacency
            )

            outputs.append(
                Ht
            )

        H = torch.stack(
            outputs,
            dim=1
        )

        # [B,T,N,D]

        return H
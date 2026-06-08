import torch
import torch.nn as nn


class TemporalSequenceBuilder(nn.Module):

    def __init__(
        self,
        graph_layer,
        dynamic_graph_builder
    ):
        super().__init__()

        self.graph_layer = graph_layer

        self.dynamic_graph_builder = (
            dynamic_graph_builder
        )

    def forward(
        self,
        x,
        cluster_matrix
    ):

        # x
        # [B,T,N,1]

        B, T, N, C = x.shape

        outputs = []

        for t in range(T):

            xt = x[:, t, :, :]

            At = self.dynamic_graph_builder(
                xt,
                cluster_matrix
            )

            # ------------------
            # FIX:
            # Use full batch graph
            # ------------------

            Ht = self.graph_layer(
                xt,
                At
            )

            outputs.append(
                Ht
            )

        H = torch.stack(
            outputs,
            dim=1
        )

        return H
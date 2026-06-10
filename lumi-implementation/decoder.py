import torch
import torch.nn as nn

from attention_layer import GraphAttentionLayer


class Decoder(nn.Module):

    def __init__(
        self,
        feature_dim=16
    ):
        super().__init__()

        self.graph_attention = (
            GraphAttentionLayer(
                in_features=feature_dim,
                out_features=feature_dim
            )
        )

    def forward(
        self,
        H,
        adjacency
    ):

        # H
        # [B,Q,N,D]

        B, Q, N, D = H.shape

        outputs = []

        for q in range(Q):

            Hq = H[:, q, :, :]

            Hq = self.graph_attention(
                Hq,
                adjacency
            )

            outputs.append(
                Hq
            )

        out = torch.stack(
            outputs,
            dim=1
        )

        # [B,Q,N,D]

        return out
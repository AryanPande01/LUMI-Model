import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_layer import GraphAttentionLayer
from dynamic_graph import DynamicGraphBuilder
from temporal_encoder import TemporalSequenceBuilder
from temporal_attention import TemporalAttention


class LUMIStage1(nn.Module):

    def __init__(
        self,
        num_nodes=542,
        hidden_dim=128
    ):
        super().__init__()

        self.dynamic_graph = DynamicGraphBuilder(
            feature_dim=1,
            hidden_dim=16
        )

        self.gat = GraphAttentionLayer(
            in_features=1,
            out_features=8
        )

        self.temporal_encoder = (
            TemporalSequenceBuilder(
                self.gat,
                self.dynamic_graph
            )
        )

        self.temporal_attention = (
            TemporalAttention(
                feature_dim=8
            )
        )

        self.fc1 = nn.Linear(
            num_nodes * 8,
            hidden_dim
        )

        self.fc2 = nn.Linear(
            hidden_dim,
            num_nodes
        )

    def forward(
        self,
        x,
        cluster_matrix
    ):

        # x
        # [B,20,542,1]

        H = self.temporal_encoder(
            x,
            cluster_matrix
        )

        # H
        # [B,20,542,8]

        H = self.temporal_attention(
            H
        )

        # [B,542,8]

        batch_size = H.shape[0]

        H = H.reshape(
            batch_size,
            -1
        )

        H = F.relu(
            self.fc1(H)
        )

        H = self.fc2(H)

        return H
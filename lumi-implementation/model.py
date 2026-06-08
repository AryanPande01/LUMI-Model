import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_layer import GraphAttentionLayer


class LUMIStage1(nn.Module):

    def __init__(
        self,
        num_nodes=542,
        hidden_dim=128
    ):
        super().__init__()

        self.gat = GraphAttentionLayer(
            in_features=1,
            out_features=8
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
        semantic_graph
    ):

        # x:
        # [B,20,542,1]

        x = x[:, -1, :, :]

        # [B,542,1]

        x = self.gat(
            x,
            semantic_graph
        )

        # [B,542,8]

        batch_size = x.shape[0]

        x = x.reshape(
            batch_size,
            -1
        )

        x = F.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x
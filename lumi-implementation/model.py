import torch
import torch.nn as nn
import torch.nn.functional as F


class SemanticGraphLayer(nn.Module):

    def __init__(self, num_nodes):
        super().__init__()

        self.num_nodes = num_nodes

    def forward(self, x, adjacency):

        # x: [B, N]
        # adjacency: [N, N]

        degree = adjacency.sum(dim=1)

        degree = degree + 1e-6

        aggregated = torch.matmul(
            x,
            adjacency.T
        )

        aggregated = aggregated / degree

        return aggregated


class LUMIStage1(nn.Module):

    def __init__(
        self,
        num_nodes=542,
        hidden_dim=128
    ):
        super().__init__()

        self.graph_layer = SemanticGraphLayer(
            num_nodes
        )

        self.fc1 = nn.Linear(
            num_nodes,
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

        # x shape:
        # [B, 20, 542, 1]

        x = x.squeeze(-1)

        # Take latest day from lookback window
        x = x[:, -1, :]

        x = self.graph_layer(
            x,
            semantic_graph
        )

        x = F.relu(
            self.fc1(x)
        )

        x = self.fc2(x)

        return x
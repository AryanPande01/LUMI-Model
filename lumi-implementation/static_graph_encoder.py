import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticGraphEncoder(nn.Module):

    def __init__(
        self,
        num_nodes=542,
        hidden_dim=128
    ):
        super().__init__()

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
        node_features,
        adjacency
    ):

        # node_features
        # [B,N]

        degree = adjacency.sum(
            dim=1,
            keepdim=True
        )

        degree = degree + 1e-6

        aggregated = torch.matmul(
            node_features,
            adjacency.T
        )

        aggregated = (
            aggregated / degree.T
        )

        aggregated = F.relu(
            self.fc1(
                aggregated
            )
        )

        aggregated = self.fc2(
            aggregated
        )

        return aggregated
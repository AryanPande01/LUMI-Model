import torch
import torch.nn as nn


class StaticFusion(nn.Module):

    def __init__(
        self,
        num_nodes=542
    ):
        super().__init__()

        self.fc = nn.Linear(
            num_nodes * 2,
            num_nodes
        )

    def forward(
        self,
        industry_features,
        wiki_features
    ):

        x = torch.cat(
            [
                industry_features,
                wiki_features
            ],
            dim=1
        )

        x = self.fc(x)

        return x
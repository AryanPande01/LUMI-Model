import torch
import torch.nn as nn


class SemanticGraphBuilder(nn.Module):

    def __init__(self):
        super().__init__()

        self.alpha = nn.Parameter(
            torch.tensor(0.0)
        )

    def forward(
        self,
        industry_graph,
        wiki_graph
    ):

        lam = torch.sigmoid(
            self.alpha
        )

        industry_graph = (
            industry_graph /
            (
                industry_graph.sum(
                    dim=1,
                    keepdim=True
                ) + 1e-6
            )
        )

        wiki_graph = (
            wiki_graph /
            (
                wiki_graph.sum(
                    dim=1,
                    keepdim=True
                ) + 1e-6
            )
        )

        semantic_graph = (
            lam * industry_graph
            +
            (1.0 - lam) * wiki_graph
        )

        return semantic_graph
import torch

from static_graph_encoder import (
    StaticGraphEncoder
)

encoder = StaticGraphEncoder()

# Simulate latest stock state
x = torch.randn(
    2,
    542
)

industry_graph = torch.ones(
    542,
    542
)

wiki_graph = torch.ones(
    542,
    542
)

industry_features = encoder(
    x,
    industry_graph
)

wiki_features = encoder(
    x,
    wiki_graph
)

fusion = torch.cat(
    [
        industry_features,
        wiki_features
    ],
    dim=1
)

print(
    "Industry:",
    industry_features.shape
)

print(
    "Wiki:",
    wiki_features.shape
)

print(
    "Fusion:",
    fusion.shape
)
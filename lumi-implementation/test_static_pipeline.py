import torch

from static_graph_encoder import StaticGraphEncoder
from static_fusion import StaticFusion

encoder = StaticGraphEncoder()
fusion = StaticFusion()

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

out = fusion(
    industry_features,
    wiki_features
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
    "Final:",
    out.shape
)
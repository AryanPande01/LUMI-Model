import torch
from model import LUMIStage1

model = LUMIStage1()

x = torch.randn(
    2,      # batch
    20,     # lookback
    542,    # stocks
    1       # features
)

cluster_matrix = torch.ones(
    542,
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

out = model(
    x,
    cluster_matrix,
    industry_graph,
    wiki_graph
)

print("Output Shape:", out.shape)
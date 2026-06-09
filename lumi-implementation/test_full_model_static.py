import torch
from model import LUMIStage1

model = LUMIStage1()

x = torch.randn(
    2,
    60,
    542,
    5
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

print(out.shape)
import torch

from model import LUMI


model = LUMI(
    num_nodes=542,
    hidden_dim=16,
    horizon=12
)

x = torch.randn(
    2,
    20,
    542,
    5
)

cluster_matrix = torch.randint(
    0,
    2,
    (542, 542)
).float()

industry_graph = torch.randint(
    0,
    2,
    (542, 542)
).float()

wiki_graph = torch.randint(
    0,
    2,
    (542, 542)
).float()

out = model(
    x,
    cluster_matrix,
    industry_graph,
    wiki_graph
)

print(out.shape)
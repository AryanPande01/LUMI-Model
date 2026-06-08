import torch

from model import LUMIStage1

model = LUMIStage1()

x = torch.randn(
    2,
    60,
    542,
    1
)

cluster_matrix = torch.ones(
    542,
    542
)

out = model(
    x,
    cluster_matrix
)

print(out.shape)
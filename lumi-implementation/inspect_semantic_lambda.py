from model import LUMI
import torch

model = LUMI()

print(
    torch.sigmoid(
        model.semantic_graph.alpha
    ).item()
)
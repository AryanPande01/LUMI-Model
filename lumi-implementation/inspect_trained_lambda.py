# inspect_trained_lambda.py

from model import LUMI
import torch

model = LUMI()

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location="cpu"
    )
)

print(
    torch.sigmoid(
        model.semantic_graph.alpha
    ).item()
)
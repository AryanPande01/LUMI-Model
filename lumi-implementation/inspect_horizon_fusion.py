from model import LUMI
import torch

model = LUMI(
    num_nodes=542,
    hidden_dim=16,
    horizon=12
)

state = torch.load(
    "best_model.pth",
    map_location="cpu"
)

model.load_state_dict(state)

print(model.horizon_fusion.weights)
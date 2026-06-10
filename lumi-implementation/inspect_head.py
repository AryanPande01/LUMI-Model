from model import LUMI
import torch

device = "cpu"

model = LUMI(
    num_nodes=542,
    hidden_dim=16,
    horizon=12
)

model.load_state_dict(
    torch.load(
        "best_model.pth",
        map_location=device
    )
)

print(model.head.weight)
print(model.head.bias)
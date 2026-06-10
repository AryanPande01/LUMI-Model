# inspect_fusion_weights.py

from model import LUMI
import torch

model = LUMI()

print(
    model.horizon_fusion.weights
)
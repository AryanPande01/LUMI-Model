# inspect_horizon_weights.py

from model import LUMI

m = LUMI()

print(
    m.horizon_fusion.weights
)
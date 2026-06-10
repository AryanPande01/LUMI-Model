import torch

from prediction_attention import PredictionAttention

x = torch.randn(
    2,
    542,
    16
)

layer = PredictionAttention(
    feature_dim=16,
    horizon=12
)

out = layer(x)

print(out.shape)
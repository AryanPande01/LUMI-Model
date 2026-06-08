import torch

from static_fusion import StaticFusion

fusion = StaticFusion()

industry = torch.randn(
    2,
    542
)

wiki = torch.randn(
    2,
    542
)

out = fusion(
    industry,
    wiki
)

print(out.shape)
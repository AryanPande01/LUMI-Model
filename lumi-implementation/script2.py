import numpy as np
from dataset import Dataset
y = dataset.targets

print(y.shape)

print(
    np.isnan(y).sum()
)

print(
    np.isinf(y).sum()
)
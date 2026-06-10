# inspect_head_weights.py

from model import LUMI

model = LUMI()

print(model.head.weight)
print(model.head.bias)
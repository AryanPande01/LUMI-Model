# inspect_semantic.py

import torch

from static_graph_loader import (
    load_static_graphs
)

from semantic_graph import (
    SemanticGraphBuilder
)

device = "cpu"

industry,wiki = load_static_graphs(
    device
)

builder = SemanticGraphBuilder()

G = builder(
    industry,
    wiki
)

print(G.shape)

print("min", G.min())
print("max", G.max())

print("mean", G.mean())

print(
    "nonzero",
    (G != 0).sum()
)

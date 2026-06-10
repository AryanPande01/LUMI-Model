# inspect_graph_density.py

from static_graph_loader import load_static_graphs
import torch

industry, wiki = load_static_graphs("cpu")

print("industry density =",
      (industry != 0).sum().item() / industry.numel())

print("wiki density =",
      (wiki != 0).sum().item() / wiki.numel())
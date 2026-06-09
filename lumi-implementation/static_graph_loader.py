import torch
import numpy as np

from graph_builder import build_adjacency


def load_static_graphs(device):

    industry_graph = build_adjacency(
        "data/LSE/graph_data/industry_adjacency.csv"
    )

    wiki_graph = build_adjacency(
        "data/LSE/graph_data/wiki_adjacency.csv"
    )

    industry_graph = torch.tensor(
        industry_graph,
        dtype=torch.float32
    ).to(device)

    wiki_graph = torch.tensor(
        wiki_graph,
        dtype=torch.float32
    ).to(device)

    return (
        industry_graph,
        wiki_graph
    )
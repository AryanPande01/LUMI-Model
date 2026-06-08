from graph_builder import (
    build_adjacency
)

industry_graph = (
    build_adjacency(
        "data/LSE/graph_data/industry_adjacency.csv"
    )
)

wiki_graph = (
    build_adjacency(
        "data/LSE/graph_data/wiki_adjacency.csv"
    )
)

print(
    "Industry:",
    industry_graph.shape
)

print(
    "Wiki:",
    wiki_graph.shape
)

print(
    "Industry Connections:",
    industry_graph.sum()
)

print(
    "Wiki Connections:",
    wiki_graph.sum()
)
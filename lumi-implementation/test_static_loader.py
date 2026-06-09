# test_static_loader.py

from static_graph_loader import load_static_graphs

industry_graph, wiki_graph = (
    load_static_graphs("cpu")
)

print(industry_graph.shape)
print(wiki_graph.shape)

print(
    "Industry Non-Zero:",
    (industry_graph != 0).sum().item()
)

print(
    "Wiki Non-Zero:",
    (wiki_graph != 0).sum().item()
)
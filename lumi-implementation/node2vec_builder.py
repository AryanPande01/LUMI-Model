import numpy as np
import networkx as nx

from graph_builder import build_adjacency
from node2vec import Node2Vec


def generate_node_embeddings():

    industry_graph = build_adjacency(
        "data/LSE/graph_data/industry_adjacency.csv"
    )

    wiki_graph = build_adjacency(
        "data/LSE/graph_data/wiki_adjacency.csv"
    )

    semantic_graph = (
        industry_graph +
        wiki_graph
    )

    G = nx.Graph()

    num_nodes = semantic_graph.shape[0]

    for i in range(num_nodes):
        G.add_node(i)

    for i in range(num_nodes):
        for j in range(num_nodes):

            if semantic_graph[i][j] > 0:
                G.add_edge(i, j)

    print(
        f"Nodes: {G.number_of_nodes()}"
    )

    print(
        f"Edges: {G.number_of_edges()}"
    )

    node2vec = Node2Vec(
        G,
        dimensions=64,
        walk_length=20,
        num_walks=50,
        workers=1
    )

    model = node2vec.fit(
        window=10,
        min_count=1
    )

    embeddings = []

    for node in range(num_nodes):

        embeddings.append(
            model.wv[str(node)]
        )

    embeddings = np.array(
        embeddings,
        dtype=np.float32
    )

    print(
        "Embedding Shape:",
        embeddings.shape
    )

    np.save(
        "embeddings.npy",
        embeddings
    )

    return embeddings


if __name__ == "__main__":
    generate_node_embeddings()
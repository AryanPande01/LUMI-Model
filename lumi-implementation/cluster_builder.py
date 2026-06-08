import numpy as np

from sklearn.neighbors import NearestNeighbors


def build_cluster_matrix(
    embedding_path="embeddings.npy",
    k=10
):

    embeddings = np.load(
        embedding_path
    )

    num_nodes = embeddings.shape[0]

    knn = NearestNeighbors(
        n_neighbors=k,
        metric="cosine"
    )

    knn.fit(
        embeddings
    )

    _, indices = knn.kneighbors(
        embeddings
    )

    M = np.zeros(
        (num_nodes, num_nodes),
        dtype=np.float32
    )

    for i in range(num_nodes):

        neighbors = indices[i]

        for j in neighbors:

            M[i][j] = 1.0
            M[j][i] = 1.0

    print(
        "Cluster Matrix Shape:",
        M.shape
    )

    print(
        "Cluster Connections:",
        M.sum()
    )

    np.save(
        "cluster_matrix.npy",
        M
    )

    return M


if __name__ == "__main__":

    build_cluster_matrix()
import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_layer import GraphAttentionLayer


class SemanticGraphEncoder(nn.Module):

    def __init__(
        self,
        feature_dim=1,
        hidden_dim=8
    ):
        super().__init__()

        self.gat = GraphAttentionLayer(
            in_features=feature_dim,
            out_features=hidden_dim
        )

    def forward(
        self,
        node_features,
        semantic_graph
    ):

        out = self.gat(
            node_features,
            semantic_graph
        )

        return out
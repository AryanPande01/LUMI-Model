import torch
import torch.nn as nn
import torch.nn.functional as F

from attention_layer import GraphAttentionLayer
from dynamic_graph import DynamicGraphBuilder
from temporal_encoder import TemporalSequenceBuilder
from temporal_attention import TemporalAttention
from long_short_builder import LongShortBuilder

from static_graph_encoder import StaticGraphEncoder
from static_fusion import StaticFusion


class LUMIStage1(nn.Module):

    def __init__(
        self,
        num_nodes=542,
        hidden_dim=128
    ):
        super().__init__()

        # ------------------
        # Dynamic Graph
        # ------------------

        self.dynamic_graph = DynamicGraphBuilder(
            feature_dim=1,
            hidden_dim=16
        )

        # ------------------
        # Spatial Attention
        # ------------------

        self.gat = GraphAttentionLayer(
            in_features=1,
            out_features=8
        )

        # ------------------
        # Temporal Encoder
        # ------------------

        self.temporal_encoder = (
            TemporalSequenceBuilder(
                self.gat,
                self.dynamic_graph
            )
        )

        # ------------------
        # Temporal Attention
        # ------------------

        self.temporal_attention = (
            TemporalAttention(
                feature_dim=8
            )
        )

        # ------------------
        # Short / Long Builder
        # ------------------

        self.long_short_builder = (
            LongShortBuilder()
        )

        # ------------------
        # Static Branch
        # ------------------

        self.static_encoder = StaticGraphEncoder(
            num_nodes=num_nodes
        )

        self.static_fusion = StaticFusion(
            num_nodes=num_nodes
        )

        # ------------------
        # Dynamic + Static Fusion
        # ------------------

        self.fusion = nn.Linear(
            num_nodes * 16 + num_nodes,
            hidden_dim
        )

        self.output_layer = nn.Linear(
            hidden_dim,
            num_nodes
        )

    def forward(
        self,
        x,
        cluster_matrix,
        industry_graph,
        wiki_graph
    ):

        # ------------------
        # Build Short & Long
        # ------------------

        short_seq, long_seq = (
            self.long_short_builder(
                x
            )
        )

        # ------------------
        # Short Branch
        # ------------------

        H_short = self.temporal_encoder(
            short_seq,
            cluster_matrix
        )

        H_short = self.temporal_attention(
            H_short
        )

        # ------------------
        # Long Branch
        # ------------------

        H_long = self.temporal_encoder(
            long_seq,
            cluster_matrix
        )

        H_long = self.temporal_attention(
            H_long
        )

        # ------------------
        # Flatten Dynamic Features
        # ------------------

        batch_size = H_short.shape[0]

        H_short = H_short.reshape(
            batch_size,
            -1
        )

        H_long = H_long.reshape(
            batch_size,
            -1
        )

        # ------------------
        # Static Branch
        # ------------------

        latest_state = x[:, -1, :, 0]

        industry_features = self.static_encoder(
            latest_state,
            industry_graph
        )

        wiki_features = self.static_encoder(
            latest_state,
            wiki_graph
        )

        static_features = self.static_fusion(
            industry_features,
            wiki_features
        )

        # ------------------
        # Fusion
        # ------------------

        H = torch.cat(
            [
                H_short,
                H_long,
                static_features
            ],
            dim=1
        )

        H = F.relu(
            self.fusion(H)
        )

        H = self.output_layer(
            H
        )

        return H
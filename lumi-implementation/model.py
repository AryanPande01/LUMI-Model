import torch
import torch.nn as nn

from attention_layer import GraphAttentionLayer
from temporal_encoder import TemporalSequenceBuilder
from temporal_attention import TemporalAttention
from prediction_attention import PredictionAttention

from semantic_graph import SemanticGraphBuilder
from dynamic_graph import DynamicGraphBuilder

from long_short_builder import LongShortBuilder

from input_projection import InputProjection
from decoder import Decoder
from horizon_fusion import HorizonFusion


class LUMI(nn.Module):

    def __init__(
        self,
        num_nodes=542,
        hidden_dim=16,
        horizon=12
    ):
        super().__init__()

        self.horizon = horizon

        # ----------------------------------
        # Input Projection
        # ----------------------------------

        self.input_projection = (
            InputProjection(
                input_dim=5,
                hidden_dim=hidden_dim
            )
        )

        # ----------------------------------
        # Semantic Graph
        # ----------------------------------

        self.semantic_graph = (
            SemanticGraphBuilder()
        )

        # ----------------------------------
        # Dynamic Graph
        # ----------------------------------

        self.dynamic_graph = (
            DynamicGraphBuilder(
                feature_dim=hidden_dim,
                hidden_dim=32
            )
        )

        # ----------------------------------
        # Spatial Attention
        # ----------------------------------

        self.gat = GraphAttentionLayer(
            in_features=hidden_dim,
            out_features=hidden_dim
        )

        # ----------------------------------
        # Temporal Encoder
        # ----------------------------------

        self.temporal_encoder = (
            TemporalSequenceBuilder(
                self.gat
            )
        )

        # ----------------------------------
        # Long / Short Builder
        # ----------------------------------

        self.long_short_builder = (
            LongShortBuilder()
        )

        # ----------------------------------
        # Temporal Attention
        # ----------------------------------

        self.temporal_attention = (
            TemporalAttention(
                feature_dim=hidden_dim
            )
        )

        # ----------------------------------
        # Prediction Attention
        # ----------------------------------

        self.prediction_attention = (
            PredictionAttention(
                feature_dim=hidden_dim,
                horizon=horizon
            )
        )

        # ----------------------------------
        # Decoder
        # ----------------------------------

        self.decoder = Decoder(
            feature_dim=hidden_dim
        )

        # ----------------------------------
        # Stream Heads
        # ----------------------------------

        self.head = nn.Linear(
            hidden_dim,
            1
        )

        # ----------------------------------
        # Horizon Fusion
        # ----------------------------------

        self.horizon_fusion = (
            HorizonFusion(
                horizon=horizon
            )
        )

    def forward(
        self,
        x,
        cluster_matrix,
        industry_graph,
        wiki_graph
    ):

        # ==================================
        # Input Projection
        # ==================================

        x = self.input_projection(
            x
        )

        # [B,T,N,D]

        # ==================================
        # Semantic Graph
        # ==================================

        semantic_graph = (
            self.semantic_graph(
                industry_graph,
                wiki_graph
            )
        )

        # ==================================
        # Dynamic Graph
        # ==================================

        latest_features = x[:, -1]

        dynamic_graph = (
            self.dynamic_graph(
                latest_features,
                cluster_matrix
            )
        )

        # ==================================
        # Long / Short
        # ==================================

        short_seq, long_seq = (
            self.long_short_builder(
                x
            )
        )

        # ==================================
        # STREAM 1
        # Semantic + Short
        # ==================================

        S1 = self.temporal_encoder(
            short_seq,
            semantic_graph
        )

        S1 = self.temporal_attention(
            S1
        )

        S1 = self.prediction_attention(
            S1
        )

        S1 = self.decoder(
            S1,
            semantic_graph
        )

        Y1 = self.head(
            S1
        ).squeeze(-1)

        # ==================================
        # STREAM 2
        # Semantic + Long
        # ==================================

        S2 = self.temporal_encoder(
            long_seq,
            semantic_graph
        )

        S2 = self.temporal_attention(
            S2
        )

        S2 = self.prediction_attention(
            S2
        )

        S2 = self.decoder(
            S2,
            semantic_graph
        )

        Y2 = self.head(
            S2
        ).squeeze(-1)

        # ==================================
        # STREAM 3
        # Dynamic + Short
        # ==================================

        S3 = self.temporal_encoder(
            short_seq,
            dynamic_graph
        )

        S3 = self.temporal_attention(
            S3
        )

        S3 = self.prediction_attention(
            S3
        )

        S3 = self.decoder(
            S3,
            dynamic_graph
        )

        Y3 = self.head(
            S3
        ).squeeze(-1)

        # ==================================
        # STREAM 4
        # Dynamic + Long
        # ==================================

        S4 = self.temporal_encoder(
            long_seq,
            dynamic_graph
        )

        S4 = self.temporal_attention(
            S4
        )

        S4 = self.prediction_attention(
            S4
        )

        S4 = self.decoder(
            S4,
            dynamic_graph
        )

        Y4 = self.head(
            S4
        ).squeeze(-1)

        # ==================================
        # Horizon-wise Fusion
        # ==================================

        out = self.horizon_fusion(
            Y1,
            Y2,
            Y3,
            Y4
        )

        # [B,Q,N]

        return out
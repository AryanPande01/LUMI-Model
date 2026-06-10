from dataset import StockDataset
from model import LUMI
from static_graph_loader import load_static_graphs
import numpy as np
import torch

device = "cpu"

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=100
)

x, _ = dataset[0]
x = x.unsqueeze(0)

cluster_matrix = torch.tensor(
    np.load("cluster_matrix.npy"),
    dtype=torch.float32
)

industry_graph, wiki_graph = load_static_graphs(device)

model = LUMI(
    num_nodes=542,
    hidden_dim=16,
    horizon=12
)

model.load_state_dict(
    torch.load(
        "final_model.pth",
        map_location=device
    )
)

model.eval()

with torch.no_grad():

    x = model.input_projection(x)

    semantic_graph = model.semantic_graph(
        industry_graph,
        wiki_graph
    )

    latest_features = x[:, -1]

    dynamic_graph = model.dynamic_graph(
        latest_features,
        cluster_matrix
    )

    short_seq, long_seq = (
        model.long_short_builder(x)
    )

    # stream 1
    S1 = model.temporal_encoder(short_seq, semantic_graph)
    S1 = model.temporal_attention(S1)
    S1 = model.prediction_attention(S1)
    S1 = model.decoder(S1, semantic_graph)
    Y1 = model.head(S1).squeeze(-1)

    # stream 2
    S2 = model.temporal_encoder(long_seq, semantic_graph)
    S2 = model.temporal_attention(S2)
    S2 = model.prediction_attention(S2)
    S2 = model.decoder(S2, semantic_graph)
    Y2 = model.head(S2).squeeze(-1)

    # stream 3
    S3 = model.temporal_encoder(short_seq, dynamic_graph)
    S3 = model.temporal_attention(S3)
    S3 = model.prediction_attention(S3)
    S3 = model.decoder(S3, dynamic_graph)
    Y3 = model.head(S3).squeeze(-1)

    # stream 4
    S4 = model.temporal_encoder(long_seq, dynamic_graph)
    S4 = model.temporal_attention(S4)
    S4 = model.prediction_attention(S4)
    S4 = model.decoder(S4, dynamic_graph)
    Y4 = model.head(S4).squeeze(-1)

    print("Y1", Y1.mean().item(), Y1.std().item())
    print("Y2", Y2.mean().item(), Y2.std().item())
    print("Y3", Y3.mean().item(), Y3.std().item())
    print("Y4", Y4.mean().item(), Y4.std().item())
    
    
out = model.horizon_fusion(
Y1,
Y2,
Y3,
Y4
)

print(
    "FINAL",
    out.mean().item(),
    out.std().item()
)
import torch

from dataset import StockDataset
from model import LUMI
from static_graph_loader import load_static_graphs

dataset = StockDataset(
    data_dir="data/LSE/data",
    lookback=60
)

x, y = dataset[0]

x = x.unsqueeze(0)

cluster_matrix = torch.load if False else None
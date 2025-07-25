# gnn/models/gnn_model.py

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
# gnn/models/gnn_model.py

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from config import config

class GNNPredictor(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.convs = torch.nn.ModuleList([
            GCNConv(in_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim),
            GCNConv(hidden_dim, hidden_dim)
        ])
        self.bns = torch.nn.ModuleList([
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim)
        ])
        self.act = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(p=0.2)
        self.fc_out = torch.nn.Linear(hidden_dim, out_dim)
        self.output_activation = torch.nn.Softplus()  # Ensures non-negative output

    def forward(self, x, edge_index):
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = self.act(x)
            x = self.drop(x)
        return self.output_activation(self.fc_out(x))  # Ensures output ≥ 0


if __name__ == "__main__":
    # Test run with dummy input
    x = torch.randn((100, 3))
    edge_index = torch.randint(0, 100, (2, 400))
    model = GNNPredictor(config["gnn_in_dim"], config["gnn_hidden_dim"], config["gnn_out_dim"])
    out = model(x, edge_index)
    print("Output shape:", out.shape)
    print("Output min value:", out.min().item())  # Should be ≥ 0


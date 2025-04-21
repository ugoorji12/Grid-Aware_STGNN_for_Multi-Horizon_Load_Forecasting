# model_definition.py
```python
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv
from torch_geometric.utils import k_hop_subgraph


def init_weights(m):
    if isinstance(m, (nn.Conv1d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class MultiScaleGATv2_LSTM(nn.Module):
    def __init__(self, node_dim, seq_dim, gat_out, gat_heads,
                 lstm_hidden, lstm_layers, edge_dim, horizon, reduced_dim=32):
        super().__init__()
        self.gat1 = GATv2Conv(node_dim, gat_out, heads=gat_heads, concat=False, edge_dim=edge_dim)
        self.gat2a = GATv2Conv(node_dim, gat_out, heads=gat_heads, concat=False, edge_dim=edge_dim)
        self.gat2b = GATv2Conv(gat_out, gat_out, heads=gat_heads, concat=False, edge_dim=edge_dim)
        self.norm = nn.LayerNorm(2*gat_out)
        self.proj = nn.Linear(2*gat_out, reduced_dim)
        self.drop = nn.Dropout(0.2)

        self.lstm = nn.LSTM(seq_dim + reduced_dim, lstm_hidden,
                            num_layers=lstm_layers, batch_first=True, dropout=0.3)
        self.out = nn.Linear(lstm_hidden, horizon)
        self.emb = None

    def precompute_embeddings(self, x, edge_index, edge_attr, node_idxs):
        one = self.drop(self.gat1(x, edge_index, edge_attr))[node_idxs]
        subset, idx2, mapping, mask = k_hop_subgraph(node_idxs, 2, edge_index,
                                                   relabel_nodes=True, flow='source_to_target')
        e2 = edge_attr[mask]
        h = self.drop(self.gat2a(x, idx2, e2))
        two = self.drop(self.gat2b(h, idx2, e2))[mapping]
        m = torch.cat((one, two), -1)
        m = self.norm(m)
        self.emb = self.proj(m).detach()

    def forward(self, seq, nodes):
        if self.emb is None:
            raise ValueError("Call precompute_embeddings first.")
        b, T, _ = seq.size()
        e = self.emb[nodes].unsqueeze(1).repeat(1, T, 1)
        inp = torch.cat((seq, e), -1)
        h, _ = self.lstm(inp)
        return self.out(h[:, -1, :])


class ModelWrapper(nn.Module):
    def __init__(self, model, default_nodes):
        super().__init__()
        self.m = model
        self.n = default_nodes

    def forward(self, x):
        return self.m(x, self.n)

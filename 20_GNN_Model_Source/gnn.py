"""Residue-graph network for variant effect regression (pure PyTorch, CPU).

A graph per structure. Node update (GraphSAGE-style mean aggregation, separate weights per edge type):
    h_{l+1} = ReLU( W_self h_l + W_seq mean_{j in N_seq(i)} h_j + W_sp mean_{j in N_sp(i)} h_j ),  dropout
The variant head reads the embedding of the mutated residue concatenated with variant-level features.
n_layers = 0 gives the matched no-graph ablation (node MLP only), with identical inputs and head.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr


def norm_adj(a):
    a = torch.as_tensor(a, dtype=torch.float32)
    d = a.sum(1, keepdim=True).clamp(min=1.0)
    return a / d


class ResidueGNN(nn.Module):
    def __init__(self, d_node, d_var, hidden=64, n_layers=2, dropout=0.2, attention=False):
        super().__init__()
        self.n_layers = n_layers
        self.inp = nn.Sequential(nn.Linear(d_node, hidden), nn.ReLU()) if d_node > 0 else None
        self.self_w = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(n_layers)])
        self.seq_w = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)])
        self.sp_w = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(n_layers)])
        self.attention = attention
        if attention:
            self.att = nn.ModuleList([nn.Linear(2 * hidden, 1) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        d_head = (hidden if d_node > 0 else 0) + d_var
        self.head = nn.Sequential(nn.Linear(d_head, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 1))

    def _att_agg(self, h, mask, l):
        n = h.shape[0]
        e = self.att[l](torch.cat([h.unsqueeze(1).expand(n, n, -1), h.unsqueeze(0).expand(n, n, -1)], -1)).squeeze(-1)
        e = e.masked_fill(mask <= 0, -1e9)
        w = torch.softmax(e, dim=1) * (mask.sum(1, keepdim=True) > 0)
        return w @ h

    def embed(self, x, a_seq, a_sp):
        h = self.inp(x)
        for l in range(self.n_layers):
            if self.attention:
                agg_sp = self._att_agg(h, a_sp, l)
            else:
                agg_sp = a_sp @ h
            h = torch.relu(self.self_w[l](h) + self.seq_w[l](a_seq @ h) + self.sp_w[l](agg_sp))
            h = self.drop(h)
        return h

    def forward(self, graphs, var_struct, var_node, var_x):
        """graphs: list of (x, a_seq, a_sp); var_struct: graph index per variant; var_node: node index."""
        if self.inp is None:
            return self.head(var_x).squeeze(-1)
        embs = [self.embed(*g) for g in graphs]
        hv = torch.zeros(len(var_x), embs[0].shape[1])
        for gi, e in enumerate(embs):
            m = var_struct == gi
            if m.any():
                hv[m] = e[var_node[m]]
        return self.head(torch.cat([hv, var_x], -1)).squeeze(-1)


def train_model(graphs, vs, vn, vx, y, train_idx, val_idx, n_layers, seed, d_node, attention=False,
                max_epochs=300, patience=30, lr=1e-3, wd=1e-4, hidden=64, dropout=0.2):
    torch.manual_seed(seed); np.random.seed(seed)
    model = ResidueGNN(d_node, vx.shape[1], hidden, n_layers, dropout, attention)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    tr = torch.as_tensor(train_idx); va = torch.as_tensor(val_idx)
    best, best_state, wait = -np.inf, None, 0
    for ep in range(max_epochs):
        model.train(); opt.zero_grad()
        pred = model(graphs, vs, vn, vx)
        loss = ((pred[tr] - y[tr]) ** 2).mean()
        loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pv = model(graphs, vs, vn, vx)[va].numpy()
        r = spearmanr(pv, y[va].numpy())[0] if len(va) > 2 else -((pv - y[va].numpy()) ** 2).mean()
        r = -1 if np.isnan(r) else r
        if r > best + 1e-4:
            best, wait = r, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, ep + 1, best

"""
pretrain_gat.py — Gate 3, part 2: self-supervised link-prediction training.

Protocol (corrected after 10-node pilot showed seed-dominated variance):
- Topology: 6 microservice namespaces x 10 services (Online-Boutique-like
  internal wiring) + cross-namespace calls. 60 nodes, ~100 dependencies.
- Proper edge split: 20% of dependencies held out for test and REMOVED
  from the message-passing graph (no structural leakage).
- Metric: AUC over held-out positives vs 3x sampled negatives.
- Baseline: MLP on raw node features (no graph). The gate criterion is
  that the GAT encoder beats the structure-blind baseline consistently,
  demonstrating the layer exploits topology.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from gat import GATEncoder


BOUTIQUE_DEPS = [(0, 1), (0, 2), (0, 3), (0, 6), (0, 7), (0, 8),
                 (1, 9), (2, 1), (2, 3), (2, 4), (2, 5), (2, 6), (2, 8),
                 (7, 6)]


def build_multi_namespace_graph(n_ns=6, seed=0):
    rng = np.random.default_rng(seed)
    deps = []
    for k in range(n_ns):
        off = 10 * k
        deps += [(a + off, b + off) for a, b in BOUTIQUE_DEPS]
    # Cross-namespace calls: each frontend calls 2 random foreign backends
    for k in range(n_ns):
        for _ in range(2):
            j = rng.integers(0, n_ns)
            while j == k:
                j = rng.integers(0, n_ns)
            deps.append((10 * k, 10 * j + int(rng.integers(1, 10))))
    deps = list(dict.fromkeys(deps))          # dedupe, keep order
    N = 10 * n_ns

    # Features: pure noise, deliberately UNcorrelated with structure.
    # (A pilot with graph-diffused features let a structure-blind MLP hit
    # AUC 0.90 from feature similarity alone -- that test measured the
    # correlation baked into the data, not structural learning. Random
    # features give the MLP nothing (~0.5), so any GAT gain above the
    # baseline is attributable to message passing over the topology.)
    X = rng.normal(0, 1, size=(N, 16))
    return torch.tensor(X, dtype=torch.float32), deps, N


def negative_edges(pos_set, N, k, rng):
    neg = set()
    while len(neg) < k:
        a, b = int(rng.integers(0, N)), int(rng.integers(0, N))
        if a != b and (a, b) not in pos_set and (a, b) not in neg:
            neg.add((a, b))
    return list(neg)


def auc_score(pos_scores, neg_scores):
    s = torch.cat([pos_scores, neg_scores])
    y = torch.cat([torch.ones_like(pos_scores),
                   torch.zeros_like(neg_scores)])
    order = s.argsort()
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(s) + 1, dtype=torch.float)
    n_pos, n_neg = len(pos_scores), len(neg_scores)
    return ((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2)
            / (n_pos * n_neg)).item()


class MLPBaseline(nn.Module):
    """Structure-blind control: same budget, raw features only."""

    def __init__(self, in_dim, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ELU(), nn.Linear(64, out_dim))

    def forward(self, x, edge_index=None):
        return self.net(x)


def run(model, x, edge_index_train, train_pos, test_pos, test_neg,
        pos_set, N, rng, epochs=1000, lr=5e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    def logits(z, pairs):
        a = torch.tensor([p[0] for p in pairs])
        b = torch.tensor([p[1] for p in pairs])
        return (z[a] * z[b]).sum(-1)

    for ep in range(epochs):
        model.train()
        z = model(x, edge_index_train)
        neg = negative_edges(pos_set, N, len(train_pos), rng)
        pl, nl = logits(z, train_pos), logits(z, neg)
        loss = (F.binary_cross_entropy_with_logits(pl, torch.ones_like(pl))
                + F.binary_cross_entropy_with_logits(nl, torch.zeros_like(nl)))
        opt.zero_grad(); loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        z = model(x, edge_index_train)
        return auc_score(logits(z, test_pos), logits(z, test_neg))


def main(seed=0):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    x, deps, N = build_multi_namespace_graph(seed=seed)
    pos_set = set(deps)

    idx = rng.permutation(len(deps))
    n_test = max(1, len(deps) // 5)
    test_pos = [deps[i] for i in idx[:n_test]]
    train_pos = [deps[i] for i in idx[n_test:]]
    test_neg = negative_edges(pos_set, N, 3 * n_test, rng)

    # Message-passing graph: TRAIN edges only, both directions.
    tr = train_pos + [(b, a) for a, b in train_pos]
    edge_index_train = torch.tensor(tr, dtype=torch.long).t()

    gat_auc = run(GATEncoder(16, 32, 32, heads=4), x, edge_index_train,
                  train_pos, test_pos, test_neg, pos_set, N, rng)
    mlp_auc = run(MLPBaseline(16), x, edge_index_train,
                  train_pos, test_pos, test_neg, pos_set, N, rng)
    return gat_auc, mlp_auc, len(deps), n_test


if __name__ == "__main__":
    gats, mlps = [], []
    for s in range(5):
        g, m, n_deps, n_test = main(seed=s)
        gats.append(g); mlps.append(m)
        print(f"seed {s}: GAT AUC {g:.3f} | MLP baseline {m:.3f}")
    print(f"\ngraph: 60 nodes, {n_deps} deps, {n_test} held out per seed")
    print(f"GAT  mean {np.mean(gats):.3f}  min {min(gats):.3f}")
    print(f"MLP  mean {np.mean(mlps):.3f}  min {min(mlps):.3f}")
    ok = np.mean(gats) > 0.75 and min(gats) > 0.6 and np.mean(gats) > np.mean(mlps) + 0.1
    print("Gate 3 part 2:", "PASSED" if ok else "FAILED",
          "(criterion: GAT mean>0.75, min>0.6, beats MLP by >0.1)")

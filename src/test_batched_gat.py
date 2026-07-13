"""Verify the batched (disjoint-graph) GAT forward equals the per-sample loop."""
import torch
from amon_agent import GraphStateEncoder
from gat import GATEncoder

torch.manual_seed(0)
enc = GraphStateEncoder(node_feat=13, glob_feat=5, embed_dim=32, heads=4)
B, N, F = 7, enc.N, 13
obs = torch.randn(B, N * F + 5)

# batched path
emb_batched, _ = enc(obs)

# reference: loop the GAT per sample
nodes = obs[:, :N*F].view(B, N, F) / enc.node_scale
ref = torch.stack([enc.gat(nodes[b], enc.edge_index) for b in range(B)], 0)

err = (emb_batched - ref).abs().max().item()
print(f"batched vs per-sample loop, max |difference|: {err:.3e}")
print("PASS" if err < 1e-5 else "FAIL")

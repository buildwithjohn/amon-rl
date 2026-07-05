"""
test_gat_equivalence.py — Gate 3 verification.

Copies weights from torch_geometric.nn.GATConv into the from-scratch
GATLayer and demands numerical agreement on fixed graphs.

PyG parameter mapping (PyG 2.8, single input tensor, add_self_loops=False,
bias=False):
    GATConv.lin.weight  -> GATLayer.W          (heads*out, in)
    GATConv.att_src     -> GATLayer.a_src      (1, heads, out) -> (heads, out)
    GATConv.att_dst     -> GATLayer.a_dst      (1, heads, out) -> (heads, out)

Run:  python -m pytest src/test_gat_equivalence.py -v
  or: python src/test_gat_equivalence.py
"""

import torch
from torch_geometric.nn import GATConv

from gat import GATLayer


def copy_weights(pyg: GATConv, mine: GATLayer):
    with torch.no_grad():
        mine.W.copy_(pyg.lin.weight)
        mine.a_src.copy_(pyg.att_src.squeeze(0))
        mine.a_dst.copy_(pyg.att_dst.squeeze(0))


def run_case(N, E, in_dim, out_dim, heads, seed, self_loops=False):
    torch.manual_seed(seed)
    x = torch.randn(N, in_dim)
    # Random directed graph; ensure every node has at least one incoming edge
    # so the softmax denominator is defined for all destinations.
    src = torch.randint(0, N, (E,))
    dst = torch.cat([torch.arange(N), torch.randint(0, N, (E - N,))])
    edge_index = torch.stack([src, dst])
    if self_loops:
        loops = torch.arange(N)
        edge_index = torch.cat(
            [edge_index, torch.stack([loops, loops])], dim=1)

    pyg = GATConv(in_dim, out_dim, heads=heads,
                  add_self_loops=False, bias=False)
    mine = GATLayer(in_dim, out_dim, heads=heads)
    copy_weights(pyg, mine)

    out_pyg = pyg(x, edge_index)
    out_mine = mine(x, edge_index)
    max_err = (out_pyg - out_mine).abs().max().item()
    return max_err


def test_small_graph_single_head():
    assert run_case(N=5, E=10, in_dim=8, out_dim=4, heads=1, seed=0) < 1e-5


def test_small_graph_multi_head():
    assert run_case(N=5, E=10, in_dim=8, out_dim=4, heads=4, seed=1) < 1e-5


def test_service_topology_scale():
    # 10 services, dense-ish dependency graph — the AMON scale.
    assert run_case(N=10, E=35, in_dim=16, out_dim=8, heads=4, seed=2) < 1e-5


def test_larger_graph_with_self_loops():
    assert run_case(N=50, E=200, in_dim=12, out_dim=6, heads=2, seed=3,
                    self_loops=True) < 1e-5


def test_gradients_flow():
    # Equivalence of forward is necessary; also confirm backward runs and
    # produces finite gradients on all parameters.
    torch.manual_seed(4)
    mine = GATLayer(8, 4, heads=2)
    x = torch.randn(6, 8, requires_grad=True)
    ei = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 2],
                       [1, 2, 3, 4, 5, 0, 3, 5]])
    out = mine(x, ei)
    out.sum().backward()
    for name, p in mine.named_parameters():
        assert p.grad is not None and torch.isfinite(p.grad).all(), name


if __name__ == "__main__":
    cases = [
        ("single head, 5 nodes", lambda: run_case(5, 10, 8, 4, 1, 0)),
        ("4 heads, 5 nodes", lambda: run_case(5, 10, 8, 4, 4, 1)),
        ("AMON scale: 10 services, 35 deps, 4 heads",
         lambda: run_case(10, 35, 16, 8, 4, 2)),
        ("50 nodes, 200 edges + self loops, 2 heads",
         lambda: run_case(50, 200, 12, 6, 2, 3, self_loops=True)),
    ]
    print("Gate 3 — GATLayer vs PyG GATConv, max |difference|:")
    ok = True
    for name, fn in cases:
        err = fn()
        status = "PASS" if err < 1e-5 else "FAIL"
        ok &= err < 1e-5
        print(f"  [{status}] {name}: {err:.2e}")
    test_gradients_flow()
    print("  [PASS] gradients finite on all parameters")
    print("\nGate 3:", "PASSED" if ok else "FAILED")

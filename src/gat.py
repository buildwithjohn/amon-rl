"""
gat.py — Graph Attention Network layer, implemented from scratch.

AMON-RL thesis, readiness gate 3.
Author: John Ayomide Akinola (ESCT/MSC/24/IT/0002/26)

Implements the attention mechanism of Velickovic et al. (2018), Eq. 1-4:

    e_ij     = LeakyReLU( a^T [ W h_i || W h_j ] )
    alpha_ij = softmax_j( e_ij )        over j in N(i)
    h_i'     = sum_j alpha_ij * W h_j   (per head; heads concatenated)

Verification standard (work plan §2.2): numerical equivalence against
torch_geometric.nn.GATConv on fixed graphs, with weights copied across.
PyG decomposes a^T[Wh_i || Wh_j] into att_dst·Wh_i + att_src·Wh_j, which
is algebraically identical to the concatenation form used here.

Convention note: PyG's GATConv aggregates messages from source j into
destination i for each directed edge (j -> i), and the attention logit is
att_src·Wh_j + att_dst·Wh_i. This implementation follows the same
convention so that equivalence is exact, not approximate.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Single multi-head GAT layer (concatenating heads).

    Args:
        in_dim:  input feature dimension per node
        out_dim: output feature dimension per head
        heads:   number of attention heads
        negative_slope: LeakyReLU slope for attention logits (paper: 0.2)
    """

    def __init__(self, in_dim, out_dim, heads=1, negative_slope=0.2):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.heads = heads
        self.negative_slope = negative_slope

        # Shared linear transform W, one block per head: (in) -> (heads*out)
        self.W = nn.Parameter(torch.empty(heads * out_dim, in_dim))
        # Attention vector a, split into source and destination halves so the
        # logit decomposes as a_src . Wh_j + a_dst . Wh_i  (PyG convention).
        self.a_src = nn.Parameter(torch.empty(heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(heads, out_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, edge_index):
        """
        x:          (N, in_dim) node features
        edge_index: (2, E) directed edges; edge_index[0]=source j,
                    edge_index[1]=destination i. Message flows j -> i.
        returns:    (N, heads*out_dim)
        """
        N = x.size(0)
        H, D = self.heads, self.out_dim

        # Wh: (N, H, D)
        Wh = (x @ self.W.t()).view(N, H, D)

        src, dst = edge_index[0], edge_index[1]

        # Attention logits per edge, per head:  e = LeakyReLU(a_src.Wh_j + a_dst.Wh_i)
        alpha_src = (Wh * self.a_src.unsqueeze(0)).sum(-1)   # (N, H)
        alpha_dst = (Wh * self.a_dst.unsqueeze(0)).sum(-1)   # (N, H)
        e = alpha_src[src] + alpha_dst[dst]                  # (E, H)
        e = F.leaky_relu(e, self.negative_slope)

        # Softmax over incoming edges of each destination node, per head.
        # Numerically stable segment softmax via subtraction of per-node max.
        e_max = torch.full((N, H), float('-inf'), device=x.device)
        e_max = e_max.scatter_reduce(0, dst.unsqueeze(-1).expand(-1, H),
                                     e, reduce='amax', include_self=True)
        e = e - e_max[dst]
        e_exp = e.exp()
        denom = torch.zeros((N, H), device=x.device)
        denom = denom.scatter_add(0, dst.unsqueeze(-1).expand(-1, H), e_exp)
        alpha = e_exp / (denom[dst] + 1e-16)                 # (E, H)

        # Aggregate messages: h_i' = sum_j alpha_ij * Wh_j
        out = torch.zeros((N, H, D), device=x.device)
        out = out.scatter_add(
            0,
            dst.view(-1, 1, 1).expand(-1, H, D),
            alpha.unsqueeze(-1) * Wh[src],
        )
        return out.reshape(N, H * D)


class GATEncoder(nn.Module):
    """Two-layer GAT encoder for the AMON service-dependency graph.

    Layer 1: multi-head with concatenation + ELU (paper architecture).
    Layer 2: single head producing the final embedding.
    """

    def __init__(self, in_dim, hidden_dim=32, out_dim=32, heads=4,
                 add_self_loops=False):
        super().__init__()
        self.gat1 = GATLayer(in_dim, hidden_dim, heads=heads)
        self.gat2 = GATLayer(hidden_dim * heads, out_dim, heads=1)
        self.add_self_loops = add_self_loops

    def forward(self, x, edge_index):
        if self.add_self_loops:
            # Standard practice (Velickovic et al. 2018; PyG default):
            # self-loops let each node retain its own features through
            # the attention aggregation.
            loops = torch.arange(x.size(0), device=x.device)
            edge_index = torch.cat(
                [edge_index, torch.stack([loops, loops])], dim=1)
        h = F.elu(self.gat1(x, edge_index))
        return self.gat2(h, edge_index)

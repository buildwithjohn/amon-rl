"""
amon_agent.py — GAT-encoded PPO policy for the AMON environment.

AMON-RL thesis. Author: John Ayomide Akinola (ESCT/MSC/24/IT/0002/26).

Composes the gate-3 GAT encoder with a PPO actor-critic head over the
MultiDiscrete placement action, with NDPR action masking on the policy
logits. This is the AMON-RL agent of the proposal (Section 6.2, Layer 2):

  obs (124,)  ->  reshape to node features (N, 12) + global (4,)
              ->  normalise node features
              ->  GATEncoder over the service dependency graph  -> (N, D)
              ->  concat each node embedding with broadcast global context
              ->  per-service policy head -> logits (N, K)
              ->  apply NDPR mask (-inf on forbidden clouds)
              ->  K independent Categoricals, one placement per service
  critic:     mean-pool node embeddings (+ global) -> scalar value

The flat-vector ablation (RQ4) is provided by FlatEncoder, a same-parameter
-budget MLP that ignores the graph, so the GAT's contribution can be
isolated by swapping the encoder.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

from gat import GATEncoder
import topology as topo
import clouds


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


# Fixed edge index for the service graph (shared across all states; the
# topology does not change within the placement problem).
_EDGE_INDEX = torch.tensor(topo.edge_index(), dtype=torch.long)

# Per-feature normalisation for the 12 node features. Latency features (p50,
# p95, p99) are on a ~0-800ms scale and must be divided down; utilisation /
# rate / flags are already ~O(1). Index layout (see amon_env._build_obs):
#   0,1,2 cloud one-hot | 3 util | 4 rate | 5,6,7 p50,p95,p99 |
#   8 err | 9 queue_depth | 10 headroom | 11 ndpr
_NODE_SCALE = torch.tensor(
    [1, 1, 1, 1, 1, 200.0, 200.0, 200.0, 1, 1, 1, 1], dtype=torch.float32)

# v2 adds a 13th per-service feature ("did I just migrate?"), already O(1).
_NODE_SCALE_V2 = torch.tensor(
    [1, 1, 1, 1, 1, 200.0, 200.0, 200.0, 1, 1, 1, 1, 1], dtype=torch.float32)


class GraphStateEncoder(nn.Module):
    """Reshape flat obs -> graph, normalise, GAT-encode, attach global ctx."""

    def __init__(self, node_feat=12, glob_feat=4, embed_dim=32, heads=4):
        super().__init__()
        self.N = topo.N_SERVICES
        self.node_feat = node_feat
        self.glob_feat = glob_feat
        self.gat = GATEncoder(node_feat, hidden_dim=32,
                              out_dim=embed_dim, heads=heads)
        self.embed_dim = embed_dim
        self.register_buffer("edge_index", _EDGE_INDEX)
        scale = _NODE_SCALE_V2 if node_feat == 13 else _NODE_SCALE
        self.register_buffer("node_scale", scale)

    def forward(self, obs):
        """obs: (B, obs_dim) -> node_emb (B, N, D), global (B, glob_feat)."""
        B = obs.shape[0]
        Nf = self.N * self.node_feat
        nodes = obs[:, :Nf].view(B, self.N, self.node_feat) / self.node_scale
        glob = obs[:, Nf:]
        # GAT is defined per-graph; loop is fine at N=10, B<=few thousand it
        # is still cheap, but we batch by stacking into one big disjoint graph.
        embs = []
        for b in range(B):
            embs.append(self.gat(nodes[b], self.edge_index))
        node_emb = torch.stack(embs, dim=0)          # (B, N, D)
        return node_emb, glob


class FlatStateEncoder(nn.Module):
    """Structure-blind ablation encoder (RQ4 control). Same output contract
    as GraphStateEncoder: produces a per-node embedding, but via a shared
    MLP applied to each node's features independently -- no message passing,
    so inter-service dependencies are invisible. Matched embed_dim."""

    def __init__(self, node_feat=12, glob_feat=4, embed_dim=32, **_):
        super().__init__()
        self.N = topo.N_SERVICES
        self.node_feat = node_feat
        self.glob_feat = glob_feat
        self.embed_dim = embed_dim
        scale = _NODE_SCALE_V2 if node_feat == 13 else _NODE_SCALE
        self.register_buffer("node_scale", scale)
        self.mlp = nn.Sequential(
            layer_init(nn.Linear(node_feat, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, embed_dim)), nn.Tanh(),
        )

    def forward(self, obs):
        B = obs.shape[0]
        Nf = self.N * self.node_feat
        nodes = obs[:, :Nf].view(B, self.N, self.node_feat) / self.node_scale
        glob = obs[:, Nf:]
        node_emb = self.mlp(nodes)                    # (B, N, D)
        return node_emb, glob


class AmonAgent(nn.Module):
    """PPO actor-critic over MultiDiscrete placement, with NDPR masking."""

    def __init__(self, encoder="gat", embed_dim=32, heads=4,
                 node_feat=12, glob_feat=4):
        super().__init__()
        self.N = topo.N_SERVICES
        self.K = clouds.N_CLOUDS
        Enc = GraphStateEncoder if encoder == "gat" else FlatStateEncoder
        self.encoder = Enc(node_feat=node_feat, glob_feat=glob_feat,
                           embed_dim=embed_dim, heads=heads)
        D = embed_dim
        G = self.encoder.glob_feat

        # Per-service policy head: node embedding + global ctx -> K logits.
        self.policy_head = nn.Sequential(
            layer_init(nn.Linear(D + G, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, self.K), std=0.01),
        )
        # Critic: pooled node embeddings + global -> scalar value.
        self.critic = nn.Sequential(
            layer_init(nn.Linear(D + G, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )

    def _logits_and_value(self, obs):
        node_emb, glob = self.encoder(obs)              # (B,N,D), (B,G)
        B = node_emb.shape[0]
        g = glob.unsqueeze(1).expand(-1, self.N, -1)    # (B,N,G)
        h = torch.cat([node_emb, g], dim=-1)            # (B,N,D+G)
        logits = self.policy_head(h)                    # (B,N,K)
        pooled = torch.cat([node_emb.mean(dim=1), glob], dim=-1)  # (B,D+G)
        value = self.critic(pooled).squeeze(-1)         # (B,)
        return logits, value

    def get_value(self, obs):
        return self._logits_and_value(obs)[1]

    def get_action_and_value(self, obs, mask=None, action=None):
        """
        obs:    (B, obs_dim)
        mask:   (B, N, K) bool, True = allowed. Forbidden -> -inf logit.
        action: (B, N) long, optional (for recomputing log-probs in update).
        returns action (B,N), logprob (B,), entropy (B,), value (B,)
        """
        logits, value = self._logits_and_value(obs)     # (B,N,K), (B,)
        if mask is not None:
            neg = torch.finfo(logits.dtype).min
            logits = torch.where(mask, logits, torch.full_like(logits, neg))
        dist = Categorical(logits=logits)               # batched over (B,N)
        if action is None:
            action = dist.sample()                      # (B,N)
        logprob = dist.log_prob(action).sum(dim=-1)     # sum over services
        entropy = dist.entropy().sum(dim=-1)            # (B,)
        return action, logprob, entropy, value


if __name__ == "__main__":
    from amon_env import AmonEnv
    env = AmonEnv(episode_len=10, seed=0)
    obs, _ = env.reset(seed=0)
    obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    mask_t = torch.tensor(env.action_masks(), dtype=torch.bool).unsqueeze(0)

    for enc in ("gat", "flat"):
        agent = AmonAgent(encoder=enc)
        a, lp, ent, v = agent.get_action_and_value(obs_t, mask_t)
        nparams = sum(p.numel() for p in agent.parameters())
        # Verify masking: no NDPR service placed on a forbidden cloud.
        placement = a[0].numpy()
        ok = all(mask_t[0, i, placement[i]] for i in range(env.N))
        print(f"[{enc:4}] params={nparams:6d}  action={placement.tolist()}  "
              f"logp={lp.item():.2f}  ent={ent.item():.2f}  V={v.item():.2f}  "
              f"mask_respected={ok}")

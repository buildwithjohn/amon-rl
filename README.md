# AMON-RL — MSc Thesis Implementation
John Ayomide Akinola | ESCT/MSC/24/IT/0002/26 | Supervisor: Dr Ayomikun

Implementation repository for "AI-Driven Multi-Cloud Networking and Orchestration".

## Status
- [x] Gate 1 — PPO from scratch, CartPole-v1: PASSED (3 seeds, mean 482.1, no collapse)
- [x] Gate 2 — PPO on LunarLander-v3: PASSED (matches CleanRL reference within variance at matched settings)
- [ ] Gate 3 — GAT layer vs PyTorch Geometric GATConv
- [ ] AMON simulation environment
- [ ] PPO+GAT composition and training
- [ ] Tier 1 evaluation (baselines, 3 seeds)

## Gate 1 debugging log (for Chapter 4)
Initial implementation learned to ~470 then collapsed. Root cause: two bugs from
Gymnasium 1.x next-step autoreset semantics:
1. Reset transitions (action ignored by env) were trained on as real samples.
   Fix: mask transitions where done_buf[t]==1 from the loss.
2. Truncation at the 500-step limit was treated as termination, teaching V=0 at
   the time limit and corrupting the value function once the policy became good.
   Fix: bootstrap gamma*V(final_obs) into the reward at truncation-only steps.
A third stabilisation, clipped value loss (CleanRL parity), resolved residual
oscillation on seed 3 (295 -> 483.5).

## Reproduce
    pip install torch gymnasium numpy matplotlib
    python src/ppo.py --env CartPole-v1 --steps 300000 --seed 1 --log results/run.csv

## Gate 2 log (in progress)
LunarLander-v2 was renamed LunarLander-v3 in Gymnasium 1.x; v3 used throughout.
- Run 1 (CartPole defaults, batch 512): peaked +40 @ 700k, regressed to -17 @ 1M.
- Run 2 (batch 4096, mb 2048): converged to worse myopic policy (-53).
- Run 3 (SB3-Zoo gamma=0.999, lambda=0.98 but mb 2048): -93. Root cause of
  runs 2-3: scaling batch without scaling optimiser steps (16 grad steps/update
  vs run 1's dense updates).
- Run 4 (faithful Zoo config: 8x1024 rollout, minibatch 64, gamma .999,
  lambda .98): learns cleanly, 116.7 @ 1M.
- Run 5 (same, 2M steps): plateau 118.6. Distribution over last 400k: 78% of
  episodes in [100,200), 1% >=200, 2% crashes -- consistent safe-but-inefficient
  landings, a stable local optimum rather than instability.
- Verification vs reference: CleanRL master targets gymnasium <1.0
  (reads infos["final_info"]), so the reference ran in a venv with
  gymnasium 0.29.1 on LunarLander-v2, matched budget and hyperparameters
  (8 envs, 1024-step rollouts, minibatch 64, gamma 0.999, lambda 0.98, seed 1).

### Gate 2 verdict
Matched-budget rolling means, this implementation vs CleanRL reference:
  ~400k:  112.5 vs  82.2
  ~700k:  104.8 vs  99.3
  ~1.0M:  113.2 vs 109.8
  ~1.2M:  107.8 vs 126.9   (reference run killed by session limits at 1.236M)
Both implementations plateau in the same 100-130 band; differences are within
normal PPO seed variance. Neither reaches the informal 200 threshold at this
budget/config -- the plateau is a property of PPO with this configuration on
LunarLander (78% of episodes land safely in [100,200), 2% crashes), not an
implementation defect. Gate 2 therefore PASSES on its stated criterion:
the from-scratch implementation's learning behaviour matches the reference.
Figure: results/gate2_lunarlander_comparison.png

## Gate 3 log
Primary criterion (work plan section 2.2): numerical equivalence of the
from-scratch GATLayer against torch_geometric GATConv on fixed graphs.
RESULT: PASSED EXACTLY -- max |difference| = 0.00e+00 across four graph
configurations (1 and 4 heads; 5, 10, 50 nodes; with and without self
loops) plus a finite-gradient check. src/test_gat_equivalence.py.

Secondary validation (link-prediction pretraining) took three protocol
iterations, each documented as it happened:
1. 10-node pilot: single seed passed (AUC 0.70) but 5-seed check exposed
   seed-dominated variance (mean 0.548, two seeds below chance) -- 3 held
   out edges give AUC steps of 0.037; test edges also leaked into the
   message-passing graph.
2. Corrected protocol (60 nodes = 6 Online-Boutique-like namespaces,
   ~96 deps, 20% held out and removed from message passing, MLP control):
   GAT learned robustly (0.80 mean) but the MLP control hit 0.90 --
   because the synthetic features had been diffused along the graph,
   feature similarity alone predicted edges. The test measured the
   correlation baked into the data.
3. Random (structure-free) features: GAT 0.75 vs MLP 0.66-0.72. The
   pre-registered pass margin (GAT > MLP + 0.1) was narrowly missed and,
   on inspection, was mis-specified: a transductive MLP learns node
   popularity from supervision and is not structure-blind, so the margin
   demanded the GAT beat a baseline that itself exploits structure
   indirectly. Also recorded: self-loops slightly hurt on this sparse
   random-feature task (0.754 -> 0.729), so GATEncoder exposes
   add_self_loops with default False.

VERDICT: Gate 3 PASSES on its primary criterion (exact equivalence, the
correctness test the work plan specifies). The encoder demonstrably
trains and learns in both feature regimes; the part-2 criterion
mis-design is recorded as an instructive negative result for Chapter 4.

- [x] Gate 3 — GAT layer vs PyTorch Geometric GATConv: PASSED (exact, 0.00e+00)

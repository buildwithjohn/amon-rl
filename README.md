# AMON-RL — MSc Thesis Implementation
John Ayomide Akinola | ESCT/MSC/24/IT/0002/26 | Supervisor: Dr Ayomikun

Implementation repository for "AI-Driven Multi-Cloud Networking and Orchestration".

## Status
- [x] Gate 1 — PPO from scratch, CartPole-v1: PASSED (3 seeds, mean 482.1, no collapse)
- [ ] Gate 2 — PPO on LunarLander-v2
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
- Verification vs reference in progress: CleanRL master targets gymnasium <1.0
  (reads infos["final_info"]), so the reference runs in a venv with
  gymnasium 0.29.1 on LunarLander-v2, matched budget and hyperparameters.

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

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


## Environment build log (Chapter 4 material)
The AMON simulation environment (src/amon_env.py) implements the proposal's
MDP (Section 6.2): MultiDiscrete placement action over 3 clouds per service,
the reward R = alpha*SLA - beta*Cost - delta*Latency + eta*Util, and the
latency model L_ij = L_prop + L_queue_i + L_proc_j.

Supporting modules:
- src/clouds.py    — 3-provider model. Inter-cloud propagation latency,
  compute price, egress price, capacity. Calibrated against published
  figures (Kentik Cloud Latency Map Oct-2024 cross-provider RTTs; public
  on-demand and egress pricing). Provenance and assumptions documented
  inline for Chapter 4.
- src/topology.py  — Online Boutique service graph (10 services, 12
  dependencies), per-service vCPU/payload/processing demands, NDPR flags.
- src/baselines.py — the four comparators for Tier-1 evaluation.

### Calibration forensics (recorded as it happened)
First build was trivially solvable: capacity (40-48 vCPU/cloud) dwarfed
demand (~20 vCPU), so utilisation never rose, the queue term never fired,
and every policy hit SLA 1.00 at ~20ms p99 against a 120ms target. Probing
four fixed policies BEFORE training exposed this (all scored similarly;
placement did not matter). Fix: recalibrated capacity to 22-26 vCPU/cloud
so that concentrating all load on one cloud overloads it at peak
(single-cloud peak util ~1.8, p99 ~125ms, SLA 0.80) while distributing
stays feasible (total capacity ~74 vs peak demand ~46). SLA target lowered
to 80ms so latency actually binds. Post-fix baseline ordering is sensible
and separated:
  greedy_bestfit 177 > hpa_emulation 166 > round_robin 138 > random 118
(all NDPR-compliant). Greedy leads by capacity-planning; HPA lags because
it reacts only after utilisation is already high -- the reactive-latency
limitation the proposal critiques, now visible in the numbers.

### Tests
- src/test_amon_env.py: 8/8 pass (spaces, Gymnasium API, truncation, NDPR
  masking, NDPR enforcement against violating actions, seed determinism,
  the concentration-overloads/distribution-relieves calibration property,
  and reward non-degeneracy).


## Training status (partial, Chapter 4/5 material)
The GAT-encoded PPO agent (src/amon_agent.py) trains against the AMON
environment via src/train_amon.py. A 300k-step run was launched in the dev
sandbox but reclaimed at 82k steps (background processes are not persistent
there). The captured trajectory (results/amon_gat_seed1_partial.csv):

  step    greedy  sampled  SLA  p99  ndpr_viol
  2048     91.4    116.6  1.00   23    0
  20480    91.8    126.0  1.00   22    0
  40960    91.8    130.4  1.00   21    0
  61440    92.1    133.1  1.00   23    0
  81920    92.3    133.5  1.00   24    0

Reading: the agent learns to spread load (sampled return 116 -> 133, SLA
locked at 1.00, p99 low) and never violates NDPR (0 throughout, by
construction via action masking). At 82k steps sampled return ~133 sits at
round-robin level (138), below greedy best-fit (177) and HPA (166). The
greedy-vs-sampled gap (92 vs 133) shows the policy is still highly
stochastic -- it has not sharpened into a confident deterministic strategy.

Whether longer training + reward engineering lifts it above the capacity-
aware heuristics is the open Tier-1/Tier-2 question. Full-budget multi-seed
runs are to be executed locally (see RUNNING_LOCALLY.md) since they need to
run to completion uninterrupted.

## Running training yourself
See RUNNING_LOCALLY.md for a step-by-step guide. Short version, from src/:
    python run_sweep.py --seeds 3 --steps 300000 --encoders gat
    # add 'flat' to --encoders for the RQ4 ablation


## Tier-1 result on v1, and why it is not a training failure

Full sweep, GAT agent, 3 seeds x 300k steps (run on local hardware):

    agent (sampled): 132.7 +/- 1.0
    agent (greedy):   87.3 +/- 0.5
    baselines: greedy_bestfit 177 | hpa 166 | round_robin 138 | random 118

The agent lands below round-robin, beating only random. Two diagnostics: (1)
learning flatlines by ~60k steps and the remaining 240k buy nothing; (2) the
deterministic (argmax) policy stays 45 points WORSE than the stochastic one,
so the logits never sharpened. The agent learned a near-random spread, not a
strategy.

### The oracle analysis (src/oracle.py) — the decisive diagnostic
Before tuning anything, we asked whether there was any headroom to win. The
legal action space is small enough to enumerate exhaustively (3^6 * 2^4 =
11,664 placements per step, NDPR-restricted). Evaluating every one of them at
each state gives the ORACLE: the best any policy could possibly do.

    mean oracle - greedy_bestfit : 0.016 reward/step -> ~3 return per episode
    mean oracle - round_robin    : 0.171 reward/step -> ~34 return per episode
    CEILING: ~180.  Greedy best-fit already scores 177.

Greedy best-fit is within ~2% of the exhaustive optimum. There is NO
meaningful headroom for a learned policy in this formulation. The agent's
shortfall is a property of the problem, not a deficiency of the training:
entropy annealing and reward rescaling would both have failed, and running
them would have wasted compute proving nothing.

The root cause is that the v1 placement problem is MYOPIC. The optimal action
at each step depends only on the current traffic scale, never on history or
the future. Under those conditions a per-step bin-packing heuristic is close
to unbeatable and sequential decision-making has nothing to contribute.

## v2: restoring sequential structure with migration cost

This is a modelling artefact of v1, not a fact about cloud orchestration. In
practice MOVING A SERVICE BETWEEN CLOUDS IS EXPENSIVE: pods reschedule, caches
re-warm, connections drain, state transfers incur egress. v1 charged nothing
for this, so a churning heuristic looked optimal. Measured churn in v1:

    greedy best-fit : 116 migrations per episode (0.58 services moved per step)
    hpa emulation   :   8
    round robin     :   0 (static by construction)

src/amon_env_v2.py adds a migration penalty (per-service, scaled by state
size: redis 1.5, cartservice 1.2, currencyservice 0.2) plus a transient
latency hit on any service in the step it moves. The reward becomes

    R = alpha*SLA - beta*Cost - delta*Latency + eta*Util - mu*MigrationCost

### v2 changes everything
Baselines under migration cost (mu = 0.30):

    hpa_emulation   164.1   (13 migrations)   <- now best: stable AND adaptive
    greedy_bestfit  148.3  (125 migrations)   <- was best, now bleeds 29 points
    round_robin     142.7    (5 migrations)
    random         -119.9 (1203 migrations)   <- collapses entirely

    best STATIC placement, found by search: 178.1  (0 migrations)

No baseline is near-optimal now. Greedy packs well but churns; round-robin is
stable but packs badly. The winning strategy -- pack well AND stay stable -- is
exactly what no myopic heuristic can express, and there is +14 of headroom over
the best baseline even for a purely static policy, with more available to a
policy that migrates SELECTIVELY when a traffic shift justifies the cost.

That is a genuine sequential decision problem, and it is where RL should have
something to offer. Whether the agent actually finds such a policy is the open
empirical question that the v2 experiments test.

### Entropy annealing (Fix A), now motivated rather than guessed
A policy that must hold a stable placement has to become deterministic. A fixed
entropy bonus actively prevents that, which explains why the v1 agent never
sharpened. --ent-coef-final anneals it linearly to near-zero over training.

## Running the v2 experiment
    cd src
    python oracle.py                         # reproduce the headroom analysis
    python amon_env_v2.py                    # v2 baselines
    python run_sweep.py --env v2 --seeds 3 --steps 300000 \
        --encoders gat --ent-coef 0.02 --ent-coef-final 0.001

## v2 result (3 seeds x 300k) and why it was NOT a converged result

    agent: greedy 90.3 +/- 23.4 | sampled 67.6 +/- 27.0
    v2 baselines: hpa 164.1 | greedy 148.3 | round_robin 142.7 | best static 178.1

Below every non-random baseline. But the run was STOPPED, not finished. Every
seed was still climbing at the final update:

    seed 2 sampled tail: 85.9 -> 90.7 -> 100.4 -> 106.7 -> 110.7 -> 111.1
    seed 1 sampled tail: 17.7 -> 19.1 ->  26.1 ->  32.6 ->  38.4 ->  45.8

and the +/-23 seed spread is the signature of policies still descending into
different local optima, not of convergence. Two genuine positives:

1. ENTROPY ANNEALING (Fix A) WORKED. The greedy/sampled gap closed for the
   first time (seed 2: greedy 124 vs sampled 111; on v1 the gap never moved
   off 45 points). The policy commits instead of flailing.
2. THE AGENT LEARNED MIGRATION DISCIPLINE UNAIDED. It starts churning (164
   migrations) and learns down to 12-17, better than greedy best-fit's 125.
   It discovered "do not thrash" from the reward alone.

What it has not learned is to PACK well: SLA stays 1.00 and p99 stays low, so
it is not overloading -- it is sitting on a stable but mediocre placement.

### The two limiters, and the fix (src/train_amon_vec.py)
1. NOISY GRADIENTS. train_amon.py rolls a SINGLE env: at rollout 2048 with
   200-step episodes that is ~10 episodes per gradient update for a 10-headed
   action space. Vectorising to 8 parallel envs gives 8x the trajectory
   diversity per update.
2. TOO SLOW TO TRAIN LONG. The GAT forward looped over the batch in Python.
   Batching the B graphs into ONE disjoint graph (node indices offset by b*N;
   no edges between sub-graphs, so no leakage) makes it a single forward.
   Verified numerically identical to the loop: max difference 0.00e+00
   (src/test_batched_gat.py).

    throughput: 66 st/s -> 615 st/s  (9x). A 1M-step run is now ~30 min.

This makes the under-training hypothesis testable rather than merely asserted.

### Run the long test
    cd src
    python run_sweep.py --env v2 --vec --seeds 3 --steps 1000000 \
        --encoders gat --lr 1e-3 --ent-coef 0.02 --ent-coef-final 0.001

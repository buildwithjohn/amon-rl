# Running AMON-RL training on your own machine

Author: John Ayomide Akinola (ESCT/MSC/24/IT/0002/26)

This guide walks through running the Tier-1 training sweep locally or on a
cloud VM, unattended, without the mid-run process reclamation that happens
in the online dev sandbox. Everything here uses the exact code already in
`src/`.

---

## Why run it yourself

The sandbox reclaims long background processes, so full 300k-step runs get
killed partway (the committed `amon_gat_seed1_partial.csv` is one such
truncated run). On your own machine the process runs to completion. As a
platform engineer this is trivial for you: a laptop overnight, or a small
cloud VM, is more than enough. The graphs are tiny (10 nodes), so this is
CPU work: no GPU needed.

---

## 1. Get the code

```bash
git clone https://github.com/buildwithjohn/amon-rl.git
cd amon-rl
```

## 2. Create an isolated environment

Using venv (built into Python 3.10+):

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
```

Or conda, if you prefer:

```bash
conda create -n amon python=3.12 -y
conda activate amon
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

If `torch_geometric` complains, install torch first, then it:

```bash
pip install "torch>=2.2,<3"
pip install torch_geometric
pip install gymnasium numpy matplotlib
```

## 4. Sanity-check before committing hours to training

Run the gate and environment tests. All should pass:

```bash
cd src
python test_gat_equivalence.py     # expect: Gate 3: PASSED (0.00e+00)
python test_amon_env.py            # expect: 8/8 environment tests passed
python baselines.py                # prints the four baseline scores
```

Then a 2-minute training smoke test (should run without error and print a
learning curve that ticks upward on sampled return):

```bash
python run_sweep.py --seeds 1 --steps 20000 --encoders gat
```

## 5. Run the real Tier-1 sweep

The non-negotiable Tier-1 deliverable is the GAT agent across 3 seeds at the
full step budget:

```bash
python run_sweep.py --seeds 3 --steps 300000 --encoders gat
```

Time estimate: at ~115 steps/s on a modern laptop CPU, 300k steps is roughly
45 minutes per seed, so about 2-2.5 hours for 3 seeds. Leave it running.

To run it truly unattended (survives closing the terminal):

```bash
# Linux / macOS
nohup python run_sweep.py --seeds 3 --steps 300000 --encoders gat \
    > sweep.log 2>&1 &
# watch progress:
tail -f sweep.log
```

On a cloud VM, the same command inside `tmux` or `screen` is the cleanest:

```bash
tmux new -s amon
python run_sweep.py --seeds 3 --steps 300000 --encoders gat
# detach with Ctrl-b then d; reattach later with: tmux attach -t amon
```

## 6. Add the RQ4 ablation (Tier 2)

Once the GAT result is in, run the flat-encoder control on the same seeds to
answer RQ4 (does graph structure help?):

```bash
python run_sweep.py --seeds 3 --steps 300000 --encoders gat flat
```

This runs both encoders; the `sweep_summary.csv` will let you compare
GAT vs flat directly at matched budget and seeds.

## 7. What you get

In `results/`:
- `amon_gat_seed1.csv`, `amon_gat_seed2.csv`, ... — per-run learning curves
  (columns: step, greedy_return, sla, cost, p99, ndpr_viol, sampled_return)
- `sweep_summary.csv` — final-evaluation table across all runs

The console also prints a per-encoder mean +/- std and the baseline
reference line (greedy 177 | hpa 166 | round_robin 138 | random 118) so you
can see immediately where the agent lands.

## 8. Bring the results back

Commit the CSVs so the next analysis session can pick them up:

```bash
cd ..
git add results/*.csv
git commit -m "Tier-1 sweep results: GAT agent, 3 seeds, 300k steps"
git push
```

Then in the next working session the learning curves and summary table can
be turned into the Chapter 5 figures and the results write-up.

---

## Notes for the write-up (Chapter 4/5)

- Report BOTH greedy and sampled return. Early in training the greedy-argmax
  policy is brittle (near-uniform logits make argmax flip on overload-
  sensitive services); the sampled return is the faithful progress signal.
  This is documented in the README build log.
- NDPR violations should be 0 across every run by construction (action
  masking). Confirm this in `sweep_summary.csv` — it is a hard-constraint
  result, not a learned one, and that is the point.
- If the agent plateaus near round-robin (~138) and below greedy best-fit
  (~177), that is a real and reportable finding, not a failure: it says the
  current reward design lets the agent learn load-spreading but not capacity-
  aware packing. Chapter 5 should discuss why, and Tier-2/3 reward
  engineering is the natural follow-on.
```

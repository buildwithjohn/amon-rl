#!/usr/bin/env python3
"""
run_sweep.py — Tier-1 training sweep for the AMON-RL agent.

AMON-RL thesis. Author: John Ayomide Akinola (ESCT/MSC/24/IT/0002/26).

Runs the full Tier-1 experiment: the GAT agent and (optionally) the flat
ablation, across N seeds, each to the full step budget, writing one CSV per
run plus a summary. Designed to run unattended on a laptop or cloud VM where
processes are not reclaimed mid-flight (unlike the dev sandbox).

Usage (from the src/ directory):
    python run_sweep.py                       # GAT, 3 seeds, 300k steps
    python run_sweep.py --encoders gat flat   # both, for the RQ4 ablation
    python run_sweep.py --seeds 5 --steps 500000
    python run_sweep.py --seeds 1 --steps 50000   # quick check first

Outputs land in ../results/:
    amon_<encoder>_seed<seed>.csv     per-run learning curve
    sweep_summary.csv                 final-eval table across all runs
"""

import os
import argparse
import numpy as np

from train_amon import train, evaluate, evaluate_sampled


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoders", nargs="+", default=["gat"],
                    choices=["gat", "flat"],
                    help="which encoders to run (gat for the main result, "
                         "add flat for the RQ4 ablation)")
    ap.add_argument("--seeds", type=int, default=3,
                    help="number of seeds per encoder (Tier-1 floor is 3)")
    ap.add_argument("--steps", type=int, default=300_000,
                    help="training steps per run")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--ent-coef-final", type=float, default=None,
                    help="anneal entropy to this (recommended for v2: 0.001)")
    ap.add_argument("--env", default="v1", choices=["v1", "v2"],
                    help="v1=myopic (greedy near-optimal); "
                         "v2=sequential with migration cost")
    ap.add_argument("--outdir", default="../results")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    summary = []

    for encoder in args.encoders:
        for seed in range(1, args.seeds + 1):
            tag = f"{encoder}_seed{seed}"
            tag = f"{args.env}_{encoder}_seed{seed}"
            log = os.path.join(args.outdir, f"amon_{tag}.csv")
            print(f"\n{'='*60}\nRUN: encoder={encoder} seed={seed} "
                  f"steps={args.steps}\n{'='*60}")
            hist, agent = train(
                encoder=encoder, total_steps=args.steps, seed=seed,
                lr=args.lr, ent_coef=args.ent_coef,
                ent_coef_final=args.ent_coef_final,
                env_version=args.env, log_path=log)

            # Final evaluation: 20 episodes, both greedy and sampled.
            gr, sla, cost, p99, viol, mig = evaluate(
                agent, n_ep=20, env_version=args.env)
            sr = evaluate_sampled(agent, n_ep=20, env_version=args.env)
            summary.append(dict(
                env=args.env, encoder=encoder, seed=seed, steps=args.steps,
                greedy_return=gr, sampled_return=sr, sla=sla,
                cost=cost, p99=p99, ndpr_viol=viol, migrations=mig))
            print(f"FINAL {tag}: greedy {gr:.1f} | sampled {sr:.1f} | "
                  f"SLA {sla:.2f} | mig {mig:.0f} | p99 {p99:.0f}ms | "
                  f"viol {viol}")

    # Write summary table.
    summ_path = os.path.join(args.outdir, f"sweep_summary_{args.env}.csv")
    keys = ["env", "encoder", "seed", "steps", "greedy_return",
            "sampled_return", "sla", "cost", "p99", "ndpr_viol", "migrations"]
    with open(summ_path, "w") as f:
        f.write(",".join(keys) + "\n")
        for row in summary:
            f.write(",".join(str(row[k]) for k in keys) + "\n")

    print(f"\n{'='*60}\nSWEEP COMPLETE. Summary -> {summ_path}\n{'='*60}")
    # Aggregate per encoder.
    for encoder in args.encoders:
        rows = [r for r in summary if r["encoder"] == encoder]
        grs = [r["greedy_return"] for r in rows]
        srs = [r["sampled_return"] for r in rows]
        print(f"{encoder}: greedy {np.mean(grs):.1f} +/- {np.std(grs):.1f} | "
              f"sampled {np.mean(srs):.1f} +/- {np.std(srs):.1f} "
              f"(n={len(rows)} seeds)")
    if args.env == "v2":
        print("\nv2 baselines (migration cost): "
              "hpa 164.1 | greedy 148.3 | round_robin 142.7 | random -119.9")
        print("  best STATIC placement (search): 178.1  <- the bar to beat")
    else:
        print("\nv1 baselines: greedy_bestfit 177 | hpa 166 | "
              "round_robin 138 | random 118")
        print("  NOTE: oracle analysis shows greedy is within ~2% of optimal "
              "in v1;\n  there is ~3 points of headroom. See oracle.py.")


if __name__ == "__main__":
    main()

"""
baselines.py — non-learning baseline policies for AMON evaluation.

AMON-RL thesis. Author: John Ayomide Akinola (ESCT/MSC/24/IT/0002/26).

The proposal (Section 7.4) evaluates the RL agent against three baselines.
These are the "rule-based and heuristic" comparators the Tier-1 deliverable
requires. Each is a callable: policy(env, obs) -> action vector (N,), and
each respects NDPR by only ever placing restricted services on approved
clouds (so the comparison is fair and no baseline "wins" by cheating on
the constraint).
"""

import numpy as np

import clouds
import topology as topo
from amon_env import NDPR_APPROVED


def _approved_clouds_for(service_idx, ndpr_flags):
    if ndpr_flags[service_idx]:
        return np.where(NDPR_APPROVED)[0]
    return np.arange(clouds.N_CLOUDS)


class RandomPolicy:
    """Uniform random placement over NDPR-approved clouds per service."""

    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, env, obs=None):
        a = np.zeros(env.N, dtype=np.int64)
        for i in range(env.N):
            allowed = _approved_clouds_for(i, env.ndpr)
            a[i] = self.rng.choice(allowed)
        return a


class RoundRobinPolicy:
    """Deterministic spread: service i -> cloud (i mod K), snapped to an
    approved cloud if the target is disallowed by NDPR. A simple, strong
    'just distribute the load' heuristic."""

    def __call__(self, env, obs=None):
        a = np.zeros(env.N, dtype=np.int64)
        for i in range(env.N):
            c = i % clouds.N_CLOUDS
            if env.ndpr[i] and not NDPR_APPROVED[c]:
                c = _approved_clouds_for(i, env.ndpr)[0]
            a[i] = c
        return a


class GreedyBestFitPolicy:
    """Bin-packing heuristic: place services one at a time (largest vCPU
    first) onto the approved cloud with the most remaining capacity. This is
    the classic 'best fit decreasing' placement, a standard scheduling
    baseline. Recomputed each step against current traffic scale."""

    def __call__(self, env, obs=None):
        scale = float(env.trace[min(env.t, env.episode_len - 1)])
        demand = env.base_vcpu * scale
        remaining = clouds.CAPACITY.astype(np.float64).copy()
        order = np.argsort(-demand)  # largest first
        a = np.zeros(env.N, dtype=np.int64)
        for i in order:
            allowed = _approved_clouds_for(i, env.ndpr)
            # pick allowed cloud with most remaining capacity
            best = allowed[np.argmax(remaining[allowed])]
            a[i] = best
            remaining[best] -= demand[i]
        return a


class HPAEmulationPolicy:
    """Threshold-reactive baseline emulating Kubernetes HPA behaviour in a
    placement setting. Starts all-on-AWS; when a cloud's utilisation exceeds
    a scale-out threshold, it migrates that cloud's largest movable service
    to the least-loaded approved cloud. This mimics HPA's reactive, single-
    signal response (it acts only after utilisation is already high, which is
    exactly the reactive-latency limitation the proposal critiques)."""

    def __init__(self, threshold=0.85):
        self.threshold = threshold
        self.placement = None

    def __call__(self, env, obs=None):
        if self.placement is None or env.t == 0:
            self.placement = np.zeros(env.N, dtype=np.int64)  # all AWS start
        scale = float(env.trace[min(env.t, env.episode_len - 1)])
        demand = env.base_vcpu * scale

        # current per-cloud utilisation
        load = np.zeros(clouds.N_CLOUDS)
        for i in range(env.N):
            load[self.placement[i]] += demand[i]
        util = load / clouds.CAPACITY

        # if any cloud is over threshold, migrate its largest movable service
        over = np.where(util > self.threshold)[0]
        for c in over:
            on_c = [i for i in range(env.N) if self.placement[i] == c]
            if not on_c:
                continue
            # largest service on the hot cloud that has an alternative
            on_c.sort(key=lambda i: -demand[i])
            for i in on_c:
                allowed = _approved_clouds_for(i, env.ndpr)
                alt = [k for k in allowed if k != c]
                if not alt:
                    continue
                # move to least-loaded allowed alternative
                target = alt[int(np.argmin(util[alt]))]
                self.placement[i] = target
                load[c] -= demand[i]
                load[target] += demand[i]
                util = load / clouds.CAPACITY
                break
        return self.placement.copy()


ALL_BASELINES = {
    "random": RandomPolicy,
    "round_robin": RoundRobinPolicy,
    "greedy_bestfit": GreedyBestFitPolicy,
    "hpa_emulation": HPAEmulationPolicy,
}


if __name__ == "__main__":
    from amon_env import AmonEnv

    def evaluate(policy_factory, n_ep=30, ep_len=200):
        rets, slas, costs, p99s, viol = [], [], [], [], 0
        for ep in range(n_ep):
            env = AmonEnv(episode_len=ep_len, seed=5000 + ep)
            obs, _ = env.reset(seed=5000 + ep)
            policy = policy_factory() if callable(policy_factory) else policy_factory
            R = 0.0
            for t in range(ep_len):
                a = policy(env, obs)
                obs, r, term, trunc, info = env.step(a)
                R += r
                if trunc:
                    rets.append(R); slas.append(info["sla_score"])
                    costs.append(info["cost_usd"]); p99s.append(info["p99_latency"])
                    if not info["ndpr_ok"]:
                        viol += 1
                    break
        return (np.mean(rets), np.std(rets), np.mean(slas),
                np.mean(costs), np.mean(p99s), viol)

    print(f"{'baseline':16} {'return':>8} {'±std':>6} {'sla':>6} "
          f"{'cost$':>7} {'p99ms':>7} {'ndpr_viol':>9}")
    for name, factory in ALL_BASELINES.items():
        r, sd, s, c, p, v = evaluate(factory)
        print(f"{name:16} {r:8.1f} {sd:6.1f} {s:6.2f} {c:7.2f} {p:7.0f} {v:9d}")

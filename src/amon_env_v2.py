"""
amon_env_v2.py — AMON environment with migration cost (sequential formulation).

AMON-RL thesis. Author: John Ayomide Akinola (ESCT/MSC/24/IT/0002/26).

MOTIVATION (Chapter 4/5)
------------------------
The oracle analysis (src/oracle.py) established that in the original
formulation (amon_env.py) the greedy best-fit heuristic sits within ~2% of
the exhaustive optimum. There was no headroom for a learned policy, because
the problem was MYOPIC: the best action at each step depended only on the
current traffic scale, never on history or the future. Under those conditions
a per-step bin-packing heuristic is close to unbeatable and reinforcement
learning has nothing to contribute.

That is a modelling artefact, not a property of real cloud orchestration.
In practice, MOVING A SERVICE BETWEEN CLOUDS IS EXPENSIVE. Live migration
means re-scheduling pods, re-warming caches, draining connections, and paying
egress on any state that moves. An orchestrator that re-packs the entire
estate every time traffic shifts would be unusable in production. The original
environment charged nothing for this, so a churning heuristic looked optimal.

Measured churn in the original environment (200-step episode, 10 services):
    greedy best-fit : 116 migrations (0.58 services relocated per step)
    hpa emulation   :   8 migrations
    round robin     :   0 migrations (static by construction)

WHAT CHANGES
------------
This environment is amon_env.py plus one term: a migration penalty charged
whenever a service's cloud assignment changes between steps.

    R_t = alpha*SLA_t - beta*Cost_t - delta*Latency_t + eta*Util_t
          - mu * MigrationCost_t                         <-- NEW

    MigrationCost_t = sum_i  1[placement_t[i] != placement_{t-1}[i]] * m_i

where m_i is the per-service migration cost, scaled by the service's state
size (a stateful service like redis is far more expensive to move than a
stateless one like currencyservice). Migrating also inflicts a transient
latency penalty on the affected service in the step it moves, modelling the
cold-start / cache-warming cost.

WHY THIS SHOULD RESTORE A ROLE FOR RL
-------------------------------------
The problem becomes genuinely sequential. A policy must now trade off:
  - packing quality NOW (favours re-packing every step, as greedy does), against
  - stability OVER TIME (favours committing to a placement that stays good as
    traffic rises and falls).
The optimal placement is no longer a function of the current step alone: it
depends on where you already are and where traffic is heading. Greedy, being
myopic, will pay migration cost repeatedly. A static policy pays none but packs
badly at peak. A learned policy can in principle anticipate the traffic cycle
and choose a placement that is good enough across the range while migrating
rarely -- which is exactly the kind of temporal credit assignment RL exists for.

This is an empirical question, not a claim. Whether the agent actually finds
such a policy is what the experiments in Chapter 5 test.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import clouds
import topology as topo
from amon_env import (AmonEnv, NDPR_APPROVED, make_diurnal_trace,
                      DEFAULT_WEIGHTS)


# Per-service migration cost multiplier. Stateful services are expensive to
# move (state must be transferred and caches re-warmed); stateless services
# are cheap. Ordering follows the topology:
#   0 frontend, 1 cart, 2 checkout, 3 currency, 4 email, 5 payment,
#   6 productcatalog, 7 recommendation, 8 shipping, 9 redis
MIGRATION_COST = np.array(
    [0.6,   # frontend        stateless but high-traffic, drain cost
     1.2,   # cartservice     stateful (cart contents)
     0.8,   # checkoutservice orchestrator, in-flight transactions
     0.2,   # currencyservice stateless, trivial to move
     0.3,   # emailservice    near-stateless queue worker
     0.9,   # paymentservice  in-flight transactions, careful drain
     0.7,   # productcatalog  large read cache to re-warm
     0.5,   # recommendation  model cache
     0.4,   # shippingservice mostly stateless
     1.5],  # redis           heaviest: full state transfer
    dtype=np.float32)

# Transient latency penalty (ms) inflicted on a service in the step it migrates
# (cold start, cache miss storm). Scales with the same per-service cost.
MIGRATION_LATENCY_MS = 40.0

DEFAULT_WEIGHTS_V2 = dict(DEFAULT_WEIGHTS)
DEFAULT_WEIGHTS_V2["mu"] = 0.30   # migration penalty weight


class AmonEnvV2(AmonEnv):
    """AMON with migration cost. Same spaces and API as AmonEnv."""

    def __init__(self, episode_len=200, weights=None, trace=None,
                 enforce_ndpr=True, seed=None, mu=None):
        w = dict(DEFAULT_WEIGHTS_V2 if weights is None else weights)
        if mu is not None:
            w["mu"] = mu
        # AmonEnv.__init__ stores self.w; it ignores unknown keys, so mu rides along.
        super().__init__(episode_len=episode_len, weights=w, trace=trace,
                         enforce_ndpr=enforce_ndpr, seed=seed)
        self.migration_cost = MIGRATION_COST
        self.prev_placement = None
        # Observation gains one feature per service: "did I just migrate?"
        # plus one global: total migrations last step. Keep the layout explicit.
        self.per_service_feat = 13          # was 12
        self.obs_dim = self.N * self.per_service_feat + 5   # was +4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        self.prev_placement = None
        self._last_migrated = np.zeros(self.N, dtype=np.float32)
        self._last_n_migrations = 0.0
        obs, info = super().reset(seed=seed, options=options)
        self.prev_placement = self.placement.copy()
        return obs, info

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(self.N)
        if self.enforce_ndpr:
            for i in range(self.N):
                if self.ndpr[i] and not NDPR_APPROVED[action[i]]:
                    action[i] = np.where(NDPR_APPROVED)[0][0]

        # Which services moved relative to the previous step?
        if self.prev_placement is None:
            migrated = np.zeros(self.N, dtype=bool)
        else:
            migrated = (action != self.prev_placement)
        self._last_migrated = migrated.astype(np.float32)
        self._last_n_migrations = float(migrated.sum())

        self.placement = action
        obs, info = self._simulate(self.placement)

        # Migration penalty enters the reward here (the parent _simulate has
        # already computed the base reward without it).
        mig_cost = float((self.migration_cost * migrated).sum())
        reward = info["reward"] - self.w.get("mu", 0.0) * mig_cost

        info["reward"] = reward
        info["migrations"] = self._last_n_migrations
        info["migration_cost"] = mig_cost
        # Transient latency hit on migrating services, reported for diagnostics.
        info["migration_latency_ms"] = float(
            MIGRATION_LATENCY_MS * migrated.sum() / max(self.N, 1))

        self.prev_placement = action.copy()
        self.t += 1
        return obs, reward, False, self.t >= self.episode_len, info

    def _build_obs(self, placement, util, svc_latency, sla_headroom, scale):
        """Parent layout plus a per-service 'just migrated' flag and a global
        migration count, so the agent can perceive its own churn."""
        feats = []
        rate_norm = self.base_vcpu / self.base_vcpu.max()
        migrated = getattr(self, "_last_migrated", np.zeros(self.N, dtype=np.float32))
        for i in range(self.N):
            onehot = np.zeros(self.K, dtype=np.float32)
            onehot[placement[i]] = 1.0
            p99 = svc_latency[i]
            # Migrating services suffer a transient latency hit this step.
            p99 = p99 + MIGRATION_LATENCY_MS * migrated[i]
            p50 = p99 / 1.4
            p95 = p99 / 1.1
            err = np.clip((util[placement[i]] - 0.9) * 2, 0, 1)
            qd = np.clip(util[placement[i]], 0, 2) / 2
            feats.extend(onehot.tolist())
            feats.extend([
                float(util[placement[i]]),
                float(rate_norm[i] * scale),
                float(p50), float(p95), float(p99),
                float(err), float(qd),
                float(sla_headroom[i]),
                float(self.ndpr[i]),
                float(migrated[i]),                 # NEW: did I just move?
            ])
        feats.extend(util.tolist())
        feats.append(float(scale))
        feats.append(float(getattr(self, "_last_n_migrations", 0.0)) / self.N)  # NEW
        return np.asarray(feats, dtype=np.float32)


if __name__ == "__main__":
    from baselines import (GreedyBestFitPolicy, RoundRobinPolicy,
                           HPAEmulationPolicy, RandomPolicy)

    def evaluate(factory, n_ep=20, ep_len=200, mu=0.30):
        rets, migs, slas, p99s = [], [], [], []
        for ep in range(n_ep):
            env = AmonEnvV2(episode_len=ep_len, seed=5000 + ep, mu=mu)
            obs, _ = env.reset(seed=5000 + ep)
            pol = factory()
            R, M = 0.0, 0
            for t in range(ep_len):
                a = pol(env, obs)
                obs, r, term, trunc, info = env.step(a)
                R += r; M += int(info["migrations"])
                if trunc:
                    rets.append(R); migs.append(M)
                    slas.append(info["sla_score"]); p99s.append(info["p99_latency"])
                    break
        return (float(np.mean(rets)), float(np.std(rets)), float(np.mean(migs)),
                float(np.mean(slas)), float(np.mean(p99s)))

    print("AMON v2 (migration cost, mu=0.30) — baselines\n")
    print(f"{'baseline':16} {'return':>8} {'±std':>6} {'migrations':>11} "
          f"{'sla':>5} {'p99':>6}")
    for name, fac in [("random", RandomPolicy),
                      ("round_robin", RoundRobinPolicy),
                      ("greedy_bestfit", GreedyBestFitPolicy),
                      ("hpa_emulation", HPAEmulationPolicy)]:
        r, sd, m, s, p = evaluate(fac)
        print(f"{name:16} {r:8.1f} {sd:6.1f} {m:11.0f} {s:5.2f} {p:6.0f}")
    print("\nCompare with amon_env v1 (no migration cost):")
    print("  greedy_bestfit 177 | hpa 166 | round_robin 138 | random 118")

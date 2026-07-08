"""
amon_env.py — AMON multi-cloud orchestration environment (Gymnasium).

AMON-RL thesis. Author: John Ayomide Akinola (ESCT/MSC/24/IT/0002/26).

Implements the MDP of the approved proposal (Section 6.2, Layer 2) as a
Gymnasium environment. The agent places each of N microservices onto one
of three clouds (AWS / Azure / GCP) each step; the environment simulates
the resulting latency, cost, SLA compliance, and utilisation, and returns
the proposal's reward.

STATE  (per service, stacked into a flat vector; the graph structure is
        supplied separately via edge_index for the GAT encoder):
  For each service i:
    - current cloud (one-hot, 3)
    - cpu utilisation of its host cloud            (1)
    - normalised request rate on the service       (1)
    - p50 / p95 / p99 latency of the service (ms)  (3)
    - error rate                                   (1)
    - queue depth (normalised)                     (1)
    - SLA headroom (fraction, can be negative)     (1)
    - NDPR flag                                    (1)
  plus global context appended once:
    - per-cloud utilisation                        (3)
    - current traffic scale                        (1)
  => obs_dim = N * 12 + 4

ACTION  (MultiDiscrete): one placement choice in {0,1,2} per service.
  The agent chooses the full placement vector each step. NDPR-restricted
  services have non-approved clouds masked out (see action_masks()).

REWARD  (proposal Eq., Section 6.2):
    R_t = alpha * SLA_t  -  beta * Cost_t  -  delta * Latency_t  +  eta * Util_t
  with all four terms normalised to comparable scales.

LATENCY  (proposal Eq., Section 6.2):
    L_ij(t) = L_prop_ij + L_queue_i(t) + L_proc_j(t)
  propagation fixed (clouds.L_PROP); queue term grows with source-cloud
  utilisation; processing term grows with destination-service load.

The workload trace (traffic scale over time) drives non-stationarity. A
synthetic diurnal-plus-spike generator is provided; real Borg/Alibaba
traces can be substituted by supplying a `trace` array.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

import clouds
import topology as topo


# Default reward weights. Tunable; these are the Tier-1 starting point.
DEFAULT_WEIGHTS = dict(alpha=1.0, beta=0.4, delta=0.6, eta=0.2)

# Approved clouds for NDPR-restricted services. In the modelled scenario,
# AWS eu-west-1 and Azure westeurope are treated as NDPR-approved for
# personal data; GCP europe-west1 is treated as non-approved, so personal
# -data services may not be placed on GCP. (A modelling choice that gives
# the masking mechanism something non-trivial to enforce; documented in
# Chapter 4 as an assumption, not a statement about real GCP compliance.)
NDPR_APPROVED = np.array([True, True, False])  # aws, azure, gcp


def make_diurnal_trace(steps, seed=0, base=1.0, amp=0.6, spikes=3):
    """Synthetic traffic-scale trace: diurnal sine + random demand spikes."""
    rng = np.random.default_rng(seed)
    t = np.arange(steps)
    # Diurnal cycle (period = full episode), never below 0.3.
    trace = base + amp * np.sin(2 * np.pi * t / max(steps, 1) - np.pi / 2)
    trace = np.clip(trace, 0.3, None)
    # Add a few short spikes.
    for _ in range(spikes):
        c = rng.integers(0, steps)
        w = max(2, steps // 25)
        mag = rng.uniform(0.8, 1.6)
        trace += mag * np.exp(-0.5 * ((t - c) / w) ** 2)
    # Small multiplicative noise.
    trace *= rng.normal(1.0, 0.05, size=steps).clip(0.7, 1.3)
    return trace.astype(np.float32)


class AmonEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, episode_len=200, weights=None, trace=None,
                 enforce_ndpr=True, seed=None):
        super().__init__()
        self.N = topo.N_SERVICES
        self.K = clouds.N_CLOUDS
        self.episode_len = episode_len
        self.w = dict(DEFAULT_WEIGHTS if weights is None else weights)
        self.enforce_ndpr = enforce_ndpr
        self._trace_override = trace

        # Static topology tensors.
        self.base_vcpu = topo.BASE_VCPU
        self.payload_mb = topo.PAYLOAD_MB
        self.base_proc = topo.BASE_PROC_MS
        self.call_rate = topo.call_rates()          # (N, N)
        self.ndpr = topo.NDPR_PERSONAL_DATA         # (N,)
        self.edge_index = topo.edge_index()         # (2, E)

        # SLA target per service (ms) on p99 end-to-end. Calibrated (Chapter
        # 4) to be binding: an all-on-one-cloud placement breaches it under
        # peak load (p99 ~125ms), while a good distribution stays well under.
        # 80ms sits in the regime where the agent must actively manage
        # placement to stay compliant, not a target that any spread satisfies.
        self.sla_p99_target = np.full(self.N, 80.0, dtype=np.float32)

        # Spaces.
        self.per_service_feat = 12
        self.obs_dim = self.N * self.per_service_feat + 4
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([self.K] * self.N)

        self._rng = np.random.default_rng(seed)
        self.placement = None
        self.t = 0
        self.trace = None

    # ---- Gymnasium API ----------------------------------------------------
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self.t = 0
        if self._trace_override is not None:
            self.trace = np.asarray(self._trace_override, dtype=np.float32)
        else:
            self.trace = make_diurnal_trace(
                self.episode_len, seed=int(self._rng.integers(1 << 30)))
        # Initial placement: everything on AWS (a valid, NDPR-compliant start
        # since AWS is approved). The agent moves services from here.
        self.placement = np.zeros(self.N, dtype=np.int64)
        obs, _ = self._simulate(self.placement)
        return obs, {}

    def step(self, action):
        action = np.asarray(action, dtype=np.int64).reshape(self.N)
        if self.enforce_ndpr:
            # Hard safety net: force any masked-out violating choice back to
            # a compliant cloud. With correct action masking upstream this
            # never triggers, but it guarantees the invariant regardless of
            # agent behaviour.
            for i in range(self.N):
                if self.ndpr[i] and not NDPR_APPROVED[action[i]]:
                    approved = np.where(NDPR_APPROVED)[0]
                    action[i] = approved[0]
        self.placement = action
        obs, info = self._simulate(self.placement)
        reward = info["reward"]
        self.t += 1
        terminated = False
        truncated = self.t >= self.episode_len
        return obs, reward, terminated, truncated, info

    # ---- Action masking (for NDPR) ---------------------------------------
    def action_masks(self):
        """(N, K) boolean mask; False = forbidden placement.

        NDPR-restricted services may only go on approved clouds. Returned
        per service so a masked-categorical policy can zero the logits of
        forbidden clouds before sampling (proposal: hard constraint via
        action masking on the policy output).
        """
        mask = np.ones((self.N, self.K), dtype=bool)
        if self.enforce_ndpr:
            for i in range(self.N):
                if self.ndpr[i]:
                    mask[i] = NDPR_APPROVED
        return mask

    # ---- Simulation core --------------------------------------------------
    def _simulate(self, placement):
        scale = float(self.trace[min(self.t, self.episode_len - 1)])

        # Per-service load and vCPU demand at current traffic scale.
        svc_vcpu = self.base_vcpu * scale                      # (N,)

        # Per-cloud utilisation = demand placed there / capacity.
        cloud_demand = np.zeros(self.K, dtype=np.float32)
        for i in range(self.N):
            cloud_demand[placement[i]] += svc_vcpu[i]
        util = cloud_demand / clouds.CAPACITY                  # (K,) can exceed 1

        # ---- Latency model: L_ij = L_prop + L_queue_i + L_proc_j ----------
        # Queue term at source cloud grows sharply as utilisation -> 1
        # (M/M/1-style congestion), capped to keep the sim finite.
        u_clip = np.clip(util, 0, 0.99)
        queue_ms = 5.0 * (u_clip / (1.0 - u_clip))             # (K,)

        # Processing term at destination service grows with its own load.
        proc_ms = self.base_proc * (1.0 + 0.5 * scale)          # (N,)

        # End-to-end p99 latency per service = sum over its outgoing calls of
        # (prop between clouds + source queue + dest proc), weighted by call
        # rate, plus a tail inflation factor for p99 over mean.
        svc_latency = np.zeros(self.N, dtype=np.float32)
        for u in range(self.N):
            total = 0.0
            wsum = 0.0
            cu = placement[u]
            for v in range(self.N):
                r = self.call_rate[u, v]
                if r <= 0:
                    continue
                cv = placement[v]
                hop = (clouds.L_PROP[cu, cv] + queue_ms[cu] + proc_ms[v])
                total += r * hop
                wsum += r
            # services with no outgoing calls still incur their own proc time
            base = proc_ms[u] + queue_ms[cu]
            mean_lat = (total / wsum) if wsum > 0 else base
            svc_latency[u] = mean_lat * 1.4   # p99 tail factor over mean

        # ---- SLA compliance: fraction of services within p99 target -------
        within = (svc_latency <= self.sla_p99_target).astype(np.float32)
        sla_score = float(within.mean())
        sla_headroom = (self.sla_p99_target - svc_latency) / self.sla_p99_target

        # ---- Cost: compute + cross-cloud egress ---------------------------
        # Compute cost per step (treat step as a unit time slice).
        compute_usd = float(np.sum(
            clouds.COMPUTE_PRICE[placement] * svc_vcpu))
        # Egress: for each call crossing a cloud boundary, GB = calls * payload.
        egress_usd = 0.0
        for u in range(self.N):
            cu = placement[u]
            for v in range(self.N):
                r = self.call_rate[u, v]
                if r <= 0 or placement[v] == cu:
                    continue
                gb = r * scale * self.payload_mb[v] / 1024.0
                egress_usd += clouds.egress_cost(cu, placement[v], gb)
        cost_usd = compute_usd + egress_usd

        # ---- Utilisation term: reward efficient packing, penalise overload
        # Mean utilisation is good up to 1.0, but overload (>1) is bad; use a
        # term that peaks near full-but-not-over.
        overload = np.clip(util - 1.0, 0, None).sum()
        util_term = float(np.clip(util, 0, 1).mean()) - 0.5 * float(overload)

        # ---- Normalise terms to comparable scales for the reward ----------
        # Latency term: normalised excess over target, averaged, clipped.
        lat_term = float(np.clip(
            (svc_latency / self.sla_p99_target).mean(), 0, 3))
        # Cost term: normalise by a rough reference (all-AWS, unit scale).
        cost_term = cost_usd / 25.0

        w = self.w
        reward = (w["alpha"] * sla_score
                  - w["beta"] * cost_term
                  - w["delta"] * lat_term
                  + w["eta"] * util_term)

        # ---- Build observation --------------------------------------------
        obs = self._build_obs(placement, util, svc_latency, sla_headroom,
                              scale)

        info = dict(
            reward=float(reward),
            sla_score=sla_score,
            cost_usd=float(cost_usd),
            compute_usd=compute_usd,
            egress_usd=float(egress_usd),
            mean_latency=float(svc_latency.mean()),
            p99_latency=float(svc_latency.max()),
            util=util.copy(),
            overload=float(overload),
            traffic_scale=scale,
            ndpr_ok=self._ndpr_ok(placement),
        )
        return obs, info

    def _build_obs(self, placement, util, svc_latency, sla_headroom, scale):
        feats = []
        rate_norm = self.base_vcpu / self.base_vcpu.max()
        for i in range(self.N):
            onehot = np.zeros(self.K, dtype=np.float32)
            onehot[placement[i]] = 1.0
            # crude p50/p95/p99 spread around the computed latency
            p99 = svc_latency[i]
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
            ])
        feats.extend(util.tolist())
        feats.append(float(scale))
        return np.asarray(feats, dtype=np.float32)

    def _ndpr_ok(self, placement):
        for i in range(self.N):
            if self.ndpr[i] and not NDPR_APPROVED[placement[i]]:
                return False
        return True


if __name__ == "__main__":
    env = AmonEnv(episode_len=50, seed=0)
    obs, _ = env.reset(seed=0)
    print("obs_dim:", env.obs_dim, "| action space:", env.action_space)
    print("action mask shape:", env.action_masks().shape)
    print("NDPR services masked to approved clouds:")
    m = env.action_masks()
    for i in range(env.N):
        if env.ndpr[i]:
            allowed = [clouds.CLOUDS[k] for k in range(env.K) if m[i, k]]
            print(f"  {topo.SERVICE_NAMES[i]}: {allowed}")

    # Random rollout sanity check.
    total_r = 0.0
    for _ in range(50):
        a = env.action_space.sample()
        obs, r, term, trunc, info = env.step(a)
        total_r += r
        if trunc:
            break
    print(f"\nrandom-policy episode return: {total_r:.2f}")
    print("last-step info: sla={sla_score:.2f} cost=${cost_usd:.2f} "
          "p99={p99_latency:.0f}ms ndpr_ok={ndpr_ok}".format(**info))

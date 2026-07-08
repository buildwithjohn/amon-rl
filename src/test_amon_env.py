"""
test_amon_env.py — behavioural tests for the AMON environment.

AMON-RL thesis. Author: John Ayomide Akinola (ESCT/MSC/24/IT/0002/26).

These are not equivalence tests (there is no reference to match, unlike the
GAT gate). They lock in the environment's contract: shapes, the Gymnasium
API, NDPR enforcement, determinism under seeding, and the calibration
properties that make it a non-trivial control problem (concentrating load
overloads; distributing relieves it).

Run:  python -m pytest src/test_amon_env.py -v
  or: python src/test_amon_env.py
"""

import numpy as np

from amon_env import AmonEnv, NDPR_APPROVED
import topology as topo
import clouds


def test_spaces_and_shapes():
    env = AmonEnv(episode_len=50, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (env.obs_dim,)
    assert env.obs_dim == topo.N_SERVICES * 12 + 4
    assert env.action_space.nvec.tolist() == [3] * topo.N_SERVICES
    assert env.action_masks().shape == (topo.N_SERVICES, clouds.N_CLOUDS)


def test_gymnasium_api_step():
    env = AmonEnv(episode_len=10, seed=0)
    env.reset(seed=0)
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    assert isinstance(r, float)
    assert term is False
    assert obs.shape == (env.obs_dim,)
    for key in ("sla_score", "cost_usd", "p99_latency", "ndpr_ok"):
        assert key in info


def test_episode_truncates_at_length():
    env = AmonEnv(episode_len=15, seed=0)
    env.reset(seed=0)
    steps = 0
    while True:
        _, _, term, trunc, _ = env.step(env.action_space.sample())
        steps += 1
        if term or trunc:
            break
    assert steps == 15 and trunc


def test_ndpr_masks_are_correct():
    env = AmonEnv(seed=0)
    env.reset(seed=0)
    m = env.action_masks()
    for i in range(env.N):
        if env.ndpr[i]:
            # NDPR service: only approved clouds allowed.
            assert (m[i] == NDPR_APPROVED).all(), topo.SERVICE_NAMES[i]
        else:
            # Unrestricted service: all clouds allowed.
            assert m[i].all(), topo.SERVICE_NAMES[i]


def test_ndpr_enforced_even_against_violating_action():
    # Force every service onto GCP (non-approved); env must relocate the
    # NDPR-restricted ones and report compliance.
    env = AmonEnv(episode_len=5, seed=0)
    env.reset(seed=0)
    _, _, _, _, info = env.step(np.full(env.N, 2, dtype=int))
    assert info["ndpr_ok"] is True
    for i in range(env.N):
        if env.ndpr[i]:
            assert NDPR_APPROVED[env.placement[i]]


def test_determinism_under_seed():
    e1 = AmonEnv(episode_len=30, seed=7); o1, _ = e1.reset(seed=7)
    e2 = AmonEnv(episode_len=30, seed=7); o2, _ = e2.reset(seed=7)
    assert np.allclose(o1, o2)
    for _ in range(30):
        a = e1.action_space.sample()  # same rng seed -> same samples
        r1 = e1.step(a)[1]
        r2 = e2.step(a)[1]
        assert abs(r1 - r2) < 1e-9


def test_concentration_overloads_distribution_relieves():
    # The calibration property: all-on-one-cloud breaches SLA under peak
    # load, round-robin distribution does not. This is what makes placement
    # a real control problem.
    def eval_policy(policy_fn, n_ep=10, ep_len=200):
        slas, p99s = [], []
        for ep in range(n_ep):
            env = AmonEnv(episode_len=ep_len, seed=2000 + ep,
                          enforce_ndpr=False)
            env.reset(seed=2000 + ep)
            for t in range(ep_len):
                _, _, _, trunc, info = env.step(policy_fn(env, t))
                if trunc:
                    slas.append(info["sla_score"])
                    p99s.append(info["p99_latency"])
                    break
        return np.mean(slas), np.mean(p99s)

    N = topo.N_SERVICES
    conc_sla, conc_p99 = eval_policy(lambda e, t: np.zeros(N, dtype=int))
    dist_sla, dist_p99 = eval_policy(lambda e, t: np.arange(N) % 3)

    # Concentration must hurt SLA and latency relative to distribution.
    assert conc_sla < dist_sla, (conc_sla, dist_sla)
    assert conc_p99 > dist_p99, (conc_p99, dist_p99)
    # And concentration must actually breach (SLA < 1) at least sometimes.
    assert conc_sla < 1.0


def test_reward_responds_to_placement():
    # Two different placements on the same trace should give different
    # returns; the reward is not degenerate.
    N = topo.N_SERVICES
    def run(policy_fn):
        env = AmonEnv(episode_len=100, seed=42, enforce_ndpr=False)
        env.reset(seed=42)
        R = 0.0
        for t in range(100):
            _, r, _, trunc, _ = env.step(policy_fn(t))
            R += r
            if trunc:
                break
        return R
    r_conc = run(lambda t: np.zeros(N, dtype=int))
    r_dist = run(lambda t: np.arange(N) % 3)
    assert abs(r_conc - r_dist) > 1.0


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"  [PASS] {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {t.__name__}: {e}")
        except Exception as e:
            print(f"  [ERR ] {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(tests)} environment tests passed")

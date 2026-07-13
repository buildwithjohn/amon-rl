"""
train_amon_vec.py — vectorised PPO training for the AMON-RL agent.

AMON-RL thesis. Author: John Ayomide Akinola (ESCT/MSC/24/IT/0002/26).

WHY THIS EXISTS
---------------
The v2 sweep (3 seeds, 300k steps) produced 90.3 +/- 23.4, below every
non-random baseline, BUT the learning curves had not converged: every seed
was still climbing at the final update (seed 2: 85.9 -> 90.7 -> 100.4 ->
106.7 -> 110.7 -> 111.1). The run was stopped, not finished. The +/- 23
spread across seeds is the signature of a policy still descending into
different local optima, not of a converged result.

Two things were limiting it, and this module fixes both:

1. NOISY GRADIENTS. train_amon.py rolls out a SINGLE environment. At
   rollout_len=2048 with 200-step episodes that is only ~10 episodes per
   gradient update, for a 10-headed MultiDiscrete action space. The
   advantage estimates are extremely noisy. Running N environments in
   parallel gives N times the trajectory diversity per update at almost no
   extra wall-clock cost (the AMON env is cheap; the GAT forward dominates,
   and that batches).

2. TOO SHORT. With a cleaner gradient the agent can be trained for the
   budget it actually needs rather than the budget that fitted the noisy
   setup.

Everything else (clipped surrogate, GAE, clipped value loss, NDPR masking,
entropy annealing) is unchanged from the verified gate-1/2 implementation.
"""

import time
import argparse

import numpy as np
import torch
import torch.nn as nn

from amon_env import AmonEnv
from amon_env_v2 import AmonEnvV2
from amon_agent import AmonAgent
from train_amon import evaluate, evaluate_sampled, env_dims


def make_env(env_version, **kw):
    return AmonEnvV2(**kw) if env_version == "v2" else AmonEnv(**kw)


def train_vec(encoder="gat", env_version="v2", total_steps=1_000_000,
              num_envs=8, ep_len=200, rollout_len=256,
              lr=3e-4, gamma=0.99, gae_lambda=0.95, clip_eps=0.2,
              ent_coef=0.02, ent_coef_final=0.001, vf_coef=0.5,
              max_grad_norm=0.5, update_epochs=4, num_minibatches=8,
              seed=1, device="cpu", eval_every=20, log_path=None):
    """rollout_len is PER ENV: the batch per update is num_envs * rollout_len."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    nf, gf = env_dims(env_version)
    agent = AmonAgent(encoder=encoder, node_feat=nf, glob_feat=gf).to(dev)
    optim = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    # Parallel envs, each with its own seed so their traffic traces differ.
    envs = [make_env(env_version, episode_len=ep_len, seed=seed * 1000 + i)
            for i in range(num_envs)]
    obs_list, mask_list = [], []
    for i, e in enumerate(envs):
        o, _ = e.reset(seed=seed * 1000 + i)
        obs_list.append(o)
        mask_list.append(e.action_masks())
    obs = torch.tensor(np.array(obs_list), dtype=torch.float32, device=dev)
    masks = torch.tensor(np.array(mask_list), dtype=torch.bool, device=dev)

    N, K, D = envs[0].N, envs[0].K, envs[0].obs_dim
    batch_size = num_envs * rollout_len
    mb_size = batch_size // num_minibatches
    num_updates = total_steps // batch_size

    # Buffers: (rollout_len, num_envs, ...)
    obs_buf = torch.zeros((rollout_len, num_envs, D), device=dev)
    act_buf = torch.zeros((rollout_len, num_envs, N), dtype=torch.long, device=dev)
    mask_buf = torch.zeros((rollout_len, num_envs, N, K), dtype=torch.bool, device=dev)
    logp_buf = torch.zeros((rollout_len, num_envs), device=dev)
    rew_buf = torch.zeros((rollout_len, num_envs), device=dev)
    done_buf = torch.zeros((rollout_len, num_envs), device=dev)
    val_buf = torch.zeros((rollout_len, num_envs), device=dev)

    history = []
    global_step = 0
    t0 = time.time()

    for update in range(1, num_updates + 1):
        frac = 1.0 - (update - 1) / num_updates
        optim.param_groups[0]["lr"] = frac * lr
        ec = (ent_coef if ent_coef_final is None
              else ent_coef_final + frac * (ent_coef - ent_coef_final))

        # ---- Rollout (all envs stepped together) ----
        for step in range(rollout_len):
            global_step += num_envs
            obs_buf[step] = obs
            mask_buf[step] = masks

            with torch.no_grad():
                a, lp, _, v = agent.get_action_and_value(obs, masks)
            act_buf[step] = a
            logp_buf[step] = lp
            val_buf[step] = v

            a_np = a.cpu().numpy()
            next_obs, next_masks, rews, dones = [], [], [], []
            for i, e in enumerate(envs):
                o, r, term, trunc, info = e.step(a_np[i])
                d = float(term or trunc)
                if d:
                    o, _ = e.reset()
                next_obs.append(o)
                next_masks.append(e.action_masks())
                rews.append(r)
                dones.append(d)
            rew_buf[step] = torch.tensor(rews, dtype=torch.float32, device=dev)
            done_buf[step] = torch.tensor(dones, dtype=torch.float32, device=dev)
            obs = torch.tensor(np.array(next_obs), dtype=torch.float32, device=dev)
            masks = torch.tensor(np.array(next_masks), dtype=torch.bool, device=dev)

        # ---- GAE ----
        with torch.no_grad():
            next_val = agent.get_value(obs)
            adv = torch.zeros_like(rew_buf)
            lastgae = torch.zeros(num_envs, device=dev)
            for t in reversed(range(rollout_len)):
                nnt = 1.0 - done_buf[t]
                nv = next_val if t == rollout_len - 1 else val_buf[t + 1]
                delta = rew_buf[t] + gamma * nv * nnt - val_buf[t]
                lastgae = delta + gamma * gae_lambda * nnt * lastgae
                adv[t] = lastgae
            ret = adv + val_buf

        # ---- Flatten and optimise ----
        b_obs = obs_buf.reshape(-1, D)
        b_act = act_buf.reshape(-1, N)
        b_mask = mask_buf.reshape(-1, N, K)
        b_logp = logp_buf.reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        b_val = val_buf.reshape(-1)

        idx = np.arange(batch_size)
        for _ in range(update_epochs):
            np.random.shuffle(idx)
            for start in range(0, batch_size, mb_size):
                mb = idx[start:start + mb_size]
                _, newlp, ent, newv = agent.get_action_and_value(
                    b_obs[mb], b_mask[mb], b_act[mb])
                ratio = (newlp - b_logp[mb]).exp()
                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                pg_loss = torch.max(pg1, pg2).mean()

                v_unc = (newv - b_ret[mb]) ** 2
                v_c = b_val[mb] + torch.clamp(newv - b_val[mb], -clip_eps, clip_eps)
                v_c = (v_c - b_ret[mb]) ** 2
                v_loss = 0.5 * torch.max(v_unc, v_c).mean()

                loss = pg_loss - ec * ent.mean() + vf_coef * v_loss
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optim.step()

        if update % eval_every == 0 or update in (1, num_updates):
            er, es, eco, ep99, ev, emig = evaluate(
                agent, device=device, env_version=env_version)
            er_s = evaluate_sampled(agent, device=device, env_version=env_version)
            history.append((global_step, er, es, eco, ep99, ev, er_s, emig))
            print(f"upd {update:4d}/{num_updates} | step {global_step:8d} "
                  f"| greedyR {er:7.1f} | sampR {er_s:7.1f} | SLA {es:.2f} "
                  f"| mig {emig:5.0f} | p99 {ep99:5.0f}ms | viol {ev} "
                  f"| ec {ec:.4f} | {global_step/(time.time()-t0):.0f} st/s",
                  flush=True)

    if log_path:
        np.savetxt(log_path, np.array(history),
                   header="step,greedy_return,sla,cost,p99,ndpr_viol,"
                          "sampled_return,migrations",
                   delimiter=",", comments="")
    return history, agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="gat", choices=["gat", "flat"])
    ap.add_argument("--env", default="v2", choices=["v1", "v2"])
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--num-envs", type=int, default=8)
    ap.add_argument("--rollout-len", type=int, default=256)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.02)
    ap.add_argument("--ent-coef-final", type=float, default=0.001)
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    hist, agent = train_vec(
        encoder=args.encoder, env_version=args.env, total_steps=args.steps,
        num_envs=args.num_envs, rollout_len=args.rollout_len, seed=args.seed,
        lr=args.lr, ent_coef=args.ent_coef, ent_coef_final=args.ent_coef_final,
        log_path=args.log)
    f = hist[-1]
    print(f"\nFINAL: greedy {f[1]:.1f} | sampled {f[6]:.1f} | SLA {f[2]:.2f} "
          f"| migrations {f[7]:.0f} | p99 {f[4]:.0f}ms | viol {f[5]}")
    if args.env == "v2":
        print("v2 baselines: hpa 164.1 | greedy 148.3 | round_robin 142.7 | "
              "best static 178.1")

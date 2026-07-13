"""
train_amon.py — PPO training for the AMON-RL agent.

AMON-RL thesis. Author: John Ayomide Akinola (ESCT/MSC/24/IT/0002/26).

Reuses the PPO mechanics verified in gates 1-2 (clipped surrogate + GAE +
clipped value loss), adapted to:
  - the AMON MultiDiscrete environment,
  - the GAT/flat encoder agent (amon_agent.AmonAgent),
  - NDPR action masks carried through rollout and update.

Single-environment rollout (the env is cheap; vectorising the per-graph GAT
forward is future optimisation, not needed for Tier 1). Logs per-update
metrics and periodic greedy-eval returns for the learning curve.
"""

import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from amon_env import AmonEnv
from amon_env_v2 import AmonEnvV2
from amon_agent import AmonAgent


def make_env(env_version, **kw):
    """v1 = myopic (amon_env), v2 = sequential with migration cost."""
    return AmonEnvV2(**kw) if env_version == "v2" else AmonEnv(**kw)


def env_dims(env_version):
    return (13, 5) if env_version == "v2" else (12, 4)


def evaluate(agent, n_ep=10, ep_len=200, seed0=9000, device="cpu",
             env_version="v1"):
    """Greedy (argmax) evaluation return + diagnostics, NDPR-masked."""
    agent.eval()
    rets, slas, costs, p99s, viol, migs = [], [], [], [], 0, []
    with torch.no_grad():
        for ep in range(n_ep):
            env = make_env(env_version, episode_len=ep_len, seed=seed0 + ep)
            obs, _ = env.reset(seed=seed0 + ep)
            R, M = 0.0, 0
            for t in range(ep_len):
                obs_t = torch.tensor(obs, dtype=torch.float32,
                                     device=device).unsqueeze(0)
                mask = torch.tensor(env.action_masks(), dtype=torch.bool,
                                    device=device).unsqueeze(0)
                logits, _ = agent._logits_and_value(obs_t)
                neg = torch.finfo(logits.dtype).min
                logits = torch.where(mask, logits, torch.full_like(logits, neg))
                action = logits.argmax(dim=-1)[0].cpu().numpy()
                obs, r, term, trunc, info = env.step(action)
                R += r
                M += int(info.get("migrations", 0))
                if trunc:
                    rets.append(R); slas.append(info["sla_score"])
                    costs.append(info["cost_usd"]); p99s.append(info["p99_latency"])
                    migs.append(M)
                    if not info["ndpr_ok"]:
                        viol += 1
                    break
    agent.train()
    return (float(np.mean(rets)), float(np.mean(slas)),
            float(np.mean(costs)), float(np.mean(p99s)), viol,
            float(np.mean(migs)) if migs else 0.0)


def evaluate_sampled(agent, n_ep=10, ep_len=200, seed0=9000, device="cpu",
                     env_version="v1"):
    """Stochastic (sampled) evaluation return, NDPR-masked. Complements the
    greedy evaluate(): when logits are near-uniform, greedy argmax is brittle,
    so the sampled return is the more faithful progress signal early on."""
    agent.eval()
    rets = []
    with torch.no_grad():
        for ep in range(n_ep):
            env = make_env(env_version, episode_len=ep_len, seed=seed0 + ep)
            obs, _ = env.reset(seed=seed0 + ep)
            R = 0.0
            for t in range(ep_len):
                obs_t = torch.tensor(obs, dtype=torch.float32,
                                     device=device).unsqueeze(0)
                mask = torch.tensor(env.action_masks(), dtype=torch.bool,
                                    device=device).unsqueeze(0)
                a, _, _, _ = agent.get_action_and_value(obs_t, mask)
                obs, r, term, trunc, info = env.step(a[0].cpu().numpy())
                R += r
                if trunc:
                    rets.append(R); break
    agent.train()
    return float(np.mean(rets))


def train(encoder="gat", total_steps=300_000, ep_len=200,
          rollout_len=2048, lr=3e-4, gamma=0.99, gae_lambda=0.95,
          clip_eps=0.2, ent_coef=0.01, ent_coef_final=None,
          vf_coef=0.5, max_grad_norm=0.5,
          update_epochs=4, num_minibatches=8, seed=1, device="cpu",
          eval_every=10, log_path=None, env_version="v1"):
    """ent_coef_final: if set, the entropy coefficient is linearly annealed
    from ent_coef to ent_coef_final over training (Fix A). A policy that must
    hold a STABLE placement has to become deterministic; a fixed entropy bonus
    actively prevents that, which is why the v1 agent never sharpened."""
    torch.manual_seed(seed); np.random.seed(seed)
    dev = torch.device(device)

    nf, gf = env_dims(env_version)
    agent = AmonAgent(encoder=encoder, node_feat=nf, glob_feat=gf).to(dev)
    optim = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    env = make_env(env_version, episode_len=ep_len, seed=seed)
    obs, _ = env.reset(seed=seed)
    obs = torch.tensor(obs, dtype=torch.float32, device=dev)
    N, K, D = env.N, env.K, env.obs_dim

    num_updates = total_steps // rollout_len
    mb_size = rollout_len // num_minibatches

    # Buffers
    obs_buf = torch.zeros((rollout_len, D), device=dev)
    act_buf = torch.zeros((rollout_len, N), dtype=torch.long, device=dev)
    mask_buf = torch.zeros((rollout_len, N, K), dtype=torch.bool, device=dev)
    logp_buf = torch.zeros(rollout_len, device=dev)
    rew_buf = torch.zeros(rollout_len, device=dev)
    done_buf = torch.zeros(rollout_len, device=dev)
    val_buf = torch.zeros(rollout_len, device=dev)

    history = []       # (global_step, eval_return, sla, cost, p99, viol)
    global_step = 0
    t0 = time.time()

    def cur_mask():
        return torch.tensor(env.action_masks(), dtype=torch.bool, device=dev)

    next_mask = cur_mask()
    for update in range(1, num_updates + 1):
        # anneal LR
        frac = 1.0 - (update - 1) / num_updates
        optim.param_groups[0]["lr"] = frac * lr
        # anneal entropy coefficient (Fix A), if requested
        if ent_coef_final is None:
            ec = ent_coef
        else:
            ec = ent_coef_final + frac * (ent_coef - ent_coef_final)

        for step in range(rollout_len):
            global_step += 1
            obs_buf[step] = obs
            mask_buf[step] = next_mask
            with torch.no_grad():
                a, lp, _, v = agent.get_action_and_value(
                    obs.unsqueeze(0), next_mask.unsqueeze(0))
            act_buf[step] = a[0]
            logp_buf[step] = lp[0]
            val_buf[step] = v[0]

            nobs, r, term, trunc, info = env.step(a[0].cpu().numpy())
            done = float(term or trunc)
            rew_buf[step] = r
            done_buf[step] = done
            if trunc or term:
                nobs, _ = env.reset()
            obs = torch.tensor(nobs, dtype=torch.float32, device=dev)
            next_mask = cur_mask()

        # GAE
        with torch.no_grad():
            next_val = agent.get_value(obs.unsqueeze(0))[0]
            adv = torch.zeros_like(rew_buf)
            lastgae = 0.0
            for t in reversed(range(rollout_len)):
                if t == rollout_len - 1:
                    nnt = 1.0 - done_buf[t]
                    nv = next_val
                else:
                    nnt = 1.0 - done_buf[t]
                    nv = val_buf[t + 1]
                delta = rew_buf[t] + gamma * nv * nnt - val_buf[t]
                adv[t] = lastgae = delta + gamma * gae_lambda * nnt * lastgae
            ret = adv + val_buf

        # Optimise
        idx = np.arange(rollout_len)
        for _ in range(update_epochs):
            np.random.shuffle(idx)
            for start in range(0, rollout_len, mb_size):
                mb = idx[start:start + mb_size]
                _, newlp, ent, newv = agent.get_action_and_value(
                    obs_buf[mb], mask_buf[mb], act_buf[mb])
                logratio = newlp - logp_buf[mb]
                ratio = logratio.exp()
                mb_adv = adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                pg1 = -mb_adv * ratio
                pg2 = -mb_adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                pg_loss = torch.max(pg1, pg2).mean()

                v_unc = (newv - ret[mb]) ** 2
                v_clp = val_buf[mb] + torch.clamp(
                    newv - val_buf[mb], -clip_eps, clip_eps)
                v_clp = (v_clp - ret[mb]) ** 2
                v_loss = 0.5 * torch.max(v_unc, v_clp).mean()

                loss = pg_loss - ec * ent.mean() + vf_coef * v_loss
                optim.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optim.step()

        if update % eval_every == 0 or update == num_updates or update == 1:
            er, es, eco, ep99, ev, emig = evaluate(
                agent, device=device, env_version=env_version)
            er_s = evaluate_sampled(agent, device=device, env_version=env_version)
            history.append((global_step, er, es, eco, ep99, ev, er_s, emig))
            print(f"upd {update:4d}/{num_updates} | step {global_step:7d} "
                  f"| greedyR {er:7.1f} | sampR {er_s:7.1f} | SLA {es:.2f} "
                  f"| mig {emig:5.0f} | p99 {ep99:5.0f}ms | viol {ev} "
                  f"| ec {ec:.4f} | {global_step/(time.time()-t0):.0f} st/s")

    if log_path:
        np.savetxt(log_path, np.array(history),
                   header="step,greedy_return,sla,cost,p99,ndpr_viol,sampled_return,migrations",
                   delimiter=",", comments="")
    return history, agent


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", default="gat", choices=["gat", "flat"])
    ap.add_argument("--steps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--ent-coef-final", type=float, default=None,
                    help="anneal entropy coef to this value (Fix A)")
    ap.add_argument("--env", default="v1", choices=["v1", "v2"],
                    help="v1=myopic, v2=sequential with migration cost")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()
    hist, agent = train(encoder=args.encoder, total_steps=args.steps,
                        seed=args.seed, lr=args.lr, ent_coef=args.ent_coef,
                        ent_coef_final=args.ent_coef_final,
                        env_version=args.env, log_path=args.log)
    final = hist[-1]
    print(f"\nfinal eval: R={final[1]:.1f} SLA={final[2]:.2f} "
          f"migrations={final[7]:.0f} p99={final[4]:.0f}ms ndpr_viol={final[5]}")

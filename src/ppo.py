"""
ppo.py — Proximal Policy Optimisation, implemented from scratch.

AMON-RL thesis, readiness gate 1.
Author: John Ayomide Akinola (ESCT/MSC/24/IT/0002/26)

Implements the clipped surrogate objective of Schulman et al. (2017),
with Generalised Advantage Estimation (Schulman et al., 2016).
Verification target: CartPole-v1 solved threshold (avg return >= 475
over 100 consecutive episodes), compared against CleanRL reference curves.

Design notes:
- Separate actor and critic MLPs (2 hidden layers, 64 units, tanh),
  matching the CleanRL reference so curves are comparable.
- Orthogonal init with gain sqrt(2) on hidden layers; policy head gain 0.01.
- Advantage normalisation per minibatch.
- No shared trunk: keeps gradients of value loss out of the policy body,
  which simplifies debugging at the cost of a few parameters.
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, 64)), nn.Tanh(),
            layer_init(nn.Linear(64, act_dim), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), self.critic(x)


def train(
    env_id="CartPole-v1",
    total_timesteps=150_000,
    num_envs=4,
    num_steps=128,          # rollout length per env
    lr=2.5e-4,
    gamma=0.99,
    gae_lambda=0.95,
    clip_eps=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    update_epochs=4,
    num_minibatches=4,
    seed=1,
    anneal_lr=True,
    log_path=None,
):
    """Train PPO. Returns list of (global_step, episodic_return) tuples."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(env_id))
         for _ in range(num_envs)]
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = envs.single_action_space.n

    agent = Agent(obs_dim, act_dim)
    optim = torch.optim.Adam(agent.parameters(), lr=lr, eps=1e-5)

    batch_size = num_envs * num_steps
    minibatch_size = batch_size // num_minibatches
    num_updates = total_timesteps // batch_size

    # Rollout buffers
    obs_buf = torch.zeros((num_steps, num_envs, obs_dim))
    act_buf = torch.zeros((num_steps, num_envs), dtype=torch.long)
    logp_buf = torch.zeros((num_steps, num_envs))
    rew_buf = torch.zeros((num_steps, num_envs))
    done_buf = torch.zeros((num_steps, num_envs))
    val_buf = torch.zeros((num_steps, num_envs))

    next_obs, _ = envs.reset(seed=seed)
    next_obs = torch.as_tensor(next_obs, dtype=torch.float32)
    next_done = torch.zeros(num_envs)

    history = []  # (global_step, episodic_return)
    global_step = 0
    t0 = time.time()

    for update in range(1, num_updates + 1):
        if anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optim.param_groups[0]["lr"] = frac * lr

        # ---- Rollout ----
        for step in range(num_steps):
            global_step += num_envs
            obs_buf[step] = next_obs
            done_buf[step] = next_done

            with torch.no_grad():
                action, logp, _, value = agent.get_action_and_value(next_obs)
            act_buf[step] = action
            logp_buf[step] = logp
            val_buf[step] = value.flatten()

            next_obs_np, reward, term, trunc, infos = envs.step(action.numpy())
            done = np.logical_or(term, trunc)
            reward = torch.as_tensor(reward, dtype=torch.float32)
            next_obs = torch.as_tensor(next_obs_np, dtype=torch.float32)

            # Gymnasium 1.x next-step autoreset, fix 2: on truncation the
            # returned obs IS the final obs; bootstrap its value so the
            # value function is not taught V=0 at the time limit.
            trunc_only = np.logical_and(trunc, np.logical_not(term))
            if trunc_only.any():
                with torch.no_grad():
                    vfinal = agent.get_value(next_obs).flatten()
                reward = reward + gamma * vfinal * torch.as_tensor(
                    trunc_only, dtype=torch.float32)

            rew_buf[step] = reward
            next_done = torch.as_tensor(done, dtype=torch.float32)

            if "episode" in infos:
                for i in range(num_envs):
                    if infos["_episode"][i]:
                        r = float(infos["episode"]["r"][i])
                        history.append((global_step, r))

        # ---- GAE ----
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()
            advantages = torch.zeros_like(rew_buf)
            lastgaelam = 0
            for t in reversed(range(num_steps)):
                if t == num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - done_buf[t + 1]
                    nextvalues = val_buf[t + 1]
                delta = rew_buf[t] + gamma * nextvalues * nextnonterminal - val_buf[t]
                advantages[t] = lastgaelam = (
                    delta + gamma * gae_lambda * nextnonterminal * lastgaelam
                )
            returns = advantages + val_buf

        # ---- Flatten batch ----
        # Gymnasium 1.x next-step autoreset, fix 1: transitions where
        # done_buf[t]==1 are reset steps whose action the env ignored;
        # exclude them from the loss.
        valid = (done_buf.reshape(-1) == 0)
        b_obs = obs_buf.reshape(-1, obs_dim)[valid]
        b_act = act_buf.reshape(-1)[valid]
        b_logp = logp_buf.reshape(-1)[valid]
        b_adv = advantages.reshape(-1)[valid]
        b_ret = returns.reshape(-1)[valid]
        b_val = val_buf.reshape(-1)[valid]
        eff_batch = int(valid.sum())

        # ---- Optimise: clipped surrogate ----
        idx = np.arange(eff_batch)
        for epoch in range(update_epochs):
            np.random.shuffle(idx)
            for start in range(0, eff_batch, minibatch_size):
                mb = idx[start:start + minibatch_size]
                _, newlogp, entropy, newval = agent.get_action_and_value(
                    b_obs[mb], b_act[mb]
                )
                logratio = newlogp - b_logp[mb]
                ratio = logratio.exp()

                mb_adv = b_adv[mb]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # L^CLIP  (Schulman et al. 2017, Eq. 7)
                pg_loss1 = -mb_adv * ratio
                pg_loss2 = -mb_adv * torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Clipped value loss (CleanRL parity)
                newval = newval.flatten()
                v_unclipped = (newval - b_ret[mb]) ** 2
                v_clipped_pred = b_val[mb] + torch.clamp(
                    newval - b_val[mb], -clip_eps, clip_eps)
                v_clipped = (v_clipped_pred - b_ret[mb]) ** 2
                v_loss = 0.5 * torch.max(v_unclipped, v_clipped).mean()
                ent_loss = entropy.mean()
                loss = pg_loss - ent_coef * ent_loss + vf_coef * v_loss

                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
                optim.step()

        if update % 20 == 0 or update == num_updates:
            recent = [r for _, r in history[-20:]]
            avg = np.mean(recent) if recent else float("nan")
            print(f"update {update:4d}/{num_updates} | step {global_step:7d} "
                  f"| avg return (last 20 eps): {avg:7.1f} "
                  f"| {global_step/(time.time()-t0):.0f} steps/s")

    envs.close()

    if log_path:
        np.savetxt(log_path,
                   np.array(history),
                   header="global_step,episodic_return",
                   delimiter=",", comments="")
    return history, agent


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="CartPole-v1")
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--log", default=None)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--num-steps", type=int, default=128)
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--num-minibatches", type=int, default=4)
    args = ap.parse_args()
    history, agent = train(env_id=args.env, total_timesteps=args.steps,
                           seed=args.seed, log_path=args.log,
                           num_envs=args.num_envs, num_steps=args.num_steps,
                           ent_coef=args.ent_coef, gamma=args.gamma,
                           gae_lambda=args.gae_lambda,
                           num_minibatches=args.num_minibatches)
    tail = [r for _, r in history[-100:]]
    print(f"\nFinal avg return over last 100 episodes: {np.mean(tail):.1f}")

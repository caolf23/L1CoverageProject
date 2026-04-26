"""Policy optimization subroutine for Algorithm 4 (neural policy PPO)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.random import Generator
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical

from models.weight_fn import TabularWeightModel

from .nn_policy import CoordPolicy, NeuralPolicy, resolve_torch_device
from .rollouts import PolicyMixture


class _CoordActorCritic(nn.Module):
    """Shared coordinate encoder with policy/value heads for PPO."""

    def __init__(self, n_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.actor = CoordPolicy(n_actions=n_actions, hidden_dim=hidden_dim)
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        h: torch.Tensor,
        *,
        rows: int,
        cols: int,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        inp = self.actor._normalize_inputs(x, y, h, rows=rows, cols=cols, horizon=horizon)
        z = F.relu(self.actor.net[0](inp))
        z = F.relu(self.actor.net[2](z))
        logits = self.actor.net[4](z)
        values = self.value_head(z).squeeze(-1)
        return logits, values


def policy_optimization_h_minus_1(
    env_factory: Any,
    cover_mixtures: list[PolicyMixture],
    w_hat: TabularWeightModel,
    *,
    layer_h: int,
    width: int,
    length: int,
    n_actions: int,
    rng: Generator,
    ppo_rollouts: int = 64,
    ppo_epochs: int = 6,
    ppo_minibatch_size: int = 128,
    ppo_clip_ratio: float = 0.2,
    ppo_lr: float = 3e-4,
    ppo_gamma: float = 1.0,
    ppo_gae_lambda: float = 0.95,
    ppo_value_coef: float = 0.5,
    ppo_entropy_coef: float = 0.01,
    ppo_max_grad_norm: float = 0.5,
    epsilon_opt: float = 0.05,
    delta_opt: float = 0.01,
    verbose: bool = False,
) -> NeuralPolicy:
    """
    ``PolicyOptimization_{h-1}(...)`` via on-policy PPO.

    This variant intentionally avoids PSDP-style prefix exploration with cover mixtures.
    Reward is sparse and only appears at 0-based timestep ``layer_h - 2``:
    ``r_t = w_hat.prob(s_t, a_t, s_{t+1})``; all other timesteps receive zero reward.
    """
    del cover_mixtures, delta_opt, epsilon_opt

    if layer_h < 2:
        raise ValueError("layer_h must be >= 2")
    device = resolve_torch_device("auto")
    ppo_t0 = time.perf_counter()

    def _to_obs(obs: np.ndarray) -> np.ndarray:
        return np.asarray(obs, dtype=np.int32)

    model = _CoordActorCritic(n_actions=n_actions, hidden_dim=64).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(ppo_lr))

    stats = {
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "entropy": 0.0,
        "approx_kl": 0.0,
        "n_transitions": 0,
        "intrinsic_return_mean": 0.0,
        "intrinsic_return_std": 0.0,
        "intrinsic_return_n_rollouts": 0,
    }
    all_episode_intrinsic_returns: list[float] = []

    for _ in range(max(int(ppo_epochs), 1)):
        obs_x: list[int] = []
        obs_y: list[int] = []
        obs_t: list[int] = []
        actions: list[int] = []
        old_logps: list[float] = []
        rewards: list[float] = []
        dones: list[float] = []
        values: list[float] = []

        ep_start_indices: list[int] = []
        ep_end_indices: list[int] = []
        for _ in range(max(int(ppo_rollouts), 1)):
            ep_start = len(obs_x)
            ep_intrinsic_return = 0.0
            env = env_factory()
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            obs = _to_obs(obs)
            terminated = truncated = False

            for t in range(layer_h - 1):
                x_i = int(obs[0])
                y_i = int(obs[1])
                h_i = int(t)
                x_ten = torch.tensor([x_i], dtype=torch.float32, device=device)
                y_ten = torch.tensor([y_i], dtype=torch.float32, device=device)
                h_ten = torch.tensor([h_i], dtype=torch.float32, device=device)
                with torch.no_grad():
                    logits, v_pred = model(
                        x_ten,
                        y_ten,
                        h_ten,
                        rows=length + 1,
                        cols=width + 1,
                        horizon=layer_h,
                    )
                    dist = Categorical(logits=logits[0])
                    act = int(dist.sample().item())
                    logp = float(dist.log_prob(torch.tensor(act, device=device)).item())
                    val = float(v_pred[0].item())

                obs_next, _env_r, terminated, truncated, _ = env.step(act)
                obs_next = _to_obs(obs_next)
                rew = 0.0
                if t == layer_h - 2:
                    rew = float(w_hat.prob(obs, act, obs_next))
                ep_intrinsic_return += rew

                obs_x.append(x_i)
                obs_y.append(y_i)
                obs_t.append(h_i)
                actions.append(act)
                old_logps.append(logp)
                rewards.append(rew)
                dones.append(1.0 if (terminated or truncated) else 0.0)
                values.append(val)

                obs = obs_next
                if terminated or truncated:
                    break
            ep_end = len(obs_x)
            if ep_end > ep_start:
                ep_start_indices.append(ep_start)
                ep_end_indices.append(ep_end)
                all_episode_intrinsic_returns.append(float(ep_intrinsic_return))

        n_steps = len(obs_x)
        if n_steps == 0:
            continue

        returns = np.zeros(n_steps, dtype=np.float32)
        adv = np.zeros(n_steps, dtype=np.float32)
        values_np = np.asarray(values, dtype=np.float32)
        rewards_np = np.asarray(rewards, dtype=np.float32)
        dones_np = np.asarray(dones, dtype=np.float32)
        gamma = float(ppo_gamma)
        lam = float(ppo_gae_lambda)
        for s_idx, e_idx in zip(ep_start_indices, ep_end_indices):
            next_value = 0.0
            gae = 0.0
            for k in range(e_idx - 1, s_idx - 1, -1):
                non_terminal = 1.0 - dones_np[k]
                delta = rewards_np[k] + gamma * next_value * non_terminal - values_np[k]
                gae = delta + gamma * lam * non_terminal * gae
                adv[k] = gae
                returns[k] = gae + values_np[k]
                next_value = values_np[k]

        adv_mean = float(np.mean(adv))
        adv_std = float(np.std(adv))
        adv = (adv - adv_mean) / max(adv_std, 1e-8)

        x_ten = torch.tensor(obs_x, dtype=torch.float32, device=device)
        y_ten = torch.tensor(obs_y, dtype=torch.float32, device=device)
        h_ten = torch.tensor(obs_t, dtype=torch.float32, device=device)
        a_ten = torch.tensor(actions, dtype=torch.long, device=device)
        old_logp_ten = torch.tensor(old_logps, dtype=torch.float32, device=device)
        ret_ten = torch.tensor(returns, dtype=torch.float32, device=device)
        adv_ten = torch.tensor(adv, dtype=torch.float32, device=device)

        batch_size = int(x_ten.shape[0])
        mb_size = max(1, min(int(ppo_minibatch_size), batch_size))
        idx = np.arange(batch_size)
        policy_loss_acc = 0.0
        value_loss_acc = 0.0
        entropy_acc = 0.0
        approx_kl_acc = 0.0
        update_steps = 0

        np.random.default_rng(int(rng.integers(0, 2**31 - 1))).shuffle(idx)
        for start in range(0, batch_size, mb_size):
            mb_idx = idx[start : start + mb_size]
            mb = torch.tensor(mb_idx, dtype=torch.long, device=device)
            logits, val_pred = model(
                x_ten[mb],
                y_ten[mb],
                h_ten[mb],
                rows=length + 1,
                cols=width + 1,
                horizon=layer_h,
            )
            dist = Categorical(logits=logits)
            new_logp = dist.log_prob(a_ten[mb])
            entropy = dist.entropy().mean()

            log_ratio = new_logp - old_logp_ten[mb]
            ratio = torch.exp(log_ratio)
            unclipped = ratio * adv_ten[mb]
            clipped = torch.clamp(
                ratio, 1.0 - float(ppo_clip_ratio), 1.0 + float(ppo_clip_ratio)
            ) * adv_ten[mb]
            policy_loss = -torch.mean(torch.minimum(unclipped, clipped))
            value_loss = F.mse_loss(val_pred, ret_ten[mb])
            loss = (
                policy_loss
                + float(ppo_value_coef) * value_loss
                - float(ppo_entropy_coef) * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(ppo_max_grad_norm))
            optimizer.step()

            with torch.no_grad():
                approx_kl = torch.mean((old_logp_ten[mb] - new_logp)).item()
            policy_loss_acc += float(policy_loss.item())
            value_loss_acc += float(value_loss.item())
            entropy_acc += float(entropy.item())
            approx_kl_acc += float(approx_kl)
            update_steps += 1

        if update_steps > 0:
            stats["policy_loss"] = policy_loss_acc / update_steps
            stats["value_loss"] = value_loss_acc / update_steps
            stats["entropy"] = entropy_acc / update_steps
            stats["approx_kl"] = approx_kl_acc / update_steps
            stats["n_transitions"] = int(n_steps)
        if all_episode_intrinsic_returns:
            ep_ret = np.asarray(all_episode_intrinsic_returns, dtype=np.float64)
            stats["intrinsic_return_mean"] = float(np.mean(ep_ret))
            stats["intrinsic_return_std"] = float(np.std(ep_ret))
            stats["intrinsic_return_n_rollouts"] = int(ep_ret.shape[0])

    policy = NeuralPolicy(
        model=model.actor,
        n_actions=n_actions,
        rows=length + 1,
        cols=width + 1,
        horizon=layer_h,
        device_preference=str(device),
    )
    policy.training_stats = {
        "policy_loss": float(stats["policy_loss"]),
        "value_loss": float(stats["value_loss"]),
        "entropy": float(stats["entropy"]),
        "approx_kl": float(stats["approx_kl"]),
        "n_transitions": int(stats["n_transitions"]),
        "intrinsic_return_mean": float(stats["intrinsic_return_mean"]),
        "intrinsic_return_std": float(stats["intrinsic_return_std"]),
        "intrinsic_return_n_rollouts": int(stats["intrinsic_return_n_rollouts"]),
        "ppo_rollouts": int(ppo_rollouts),
        "ppo_epochs": int(ppo_epochs),
    }
    if verbose:
        print(f"  [timing] ppo training: {time.perf_counter() - ppo_t0:.3f}s")
    return policy

"""Policy optimization subroutine for Algorithm 4 (neural policy PSDP)."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from numpy.random import Generator
import torch
import torch.nn.functional as F

from models.weight_fn import TabularWeightModel

from .nn_policy import CoordPolicy, NeuralPolicy, resolve_torch_device
from .rollouts import PolicyMixture


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
    psdp_samples: int = 600,
    epsilon_opt: float = 0.05,
    delta_opt: float = 0.01,
    psdp_epsilon_greedy: float = 0.05,
    verbose: bool = False,
) -> NeuralPolicy:
    """
    ``PolicyOptimization_{h-1}(...)`` via PSDP rollouts, then fit a coordinate MLP.

    For each layer ``l`` (0-based timestep) from ``h-2`` down to ``0``, collect online
    tuples ``(x_l, a_l, R_l)`` where reward uses ``w_hat`` on observed transitions and
    suffix actions follow already-computed future policies. Aggregate empirical Q-labels
    per (timestep, state), derive best-action targets, and train a neural policy with
    cross-entropy.
    """
    del delta_opt, epsilon_opt

    if layer_h < 2:
        raise ValueError("layer_h must be >= 2")
    if len(cover_mixtures) < layer_h - 1:
        raise ValueError("need p_1..p_{h-1} cover mixtures")

    epsilon = float(np.clip(psdp_epsilon_greedy, 0.0, 1.0))
    device = resolve_torch_device("auto")
    q_sums = np.zeros((layer_h - 1, length + 1, width + 1, n_actions), dtype=np.float64)
    q_counts = np.zeros((layer_h - 1, length + 1, width + 1, n_actions), dtype=np.int32)
    policy_logits = np.zeros((layer_h - 1, length + 1, width + 1, n_actions), dtype=np.float64)

    def _sample_from_future_policy(obs: np.ndarray, timestep: int) -> int:
        x = int(np.asarray(obs)[0])
        y = int(np.asarray(obs)[1])
        if timestep < 0 or timestep >= policy_logits.shape[0]:
            return int(rng.integers(0, n_actions))
        p = policy_logits[timestep, x, y]
        if np.max(p) <= 0.0:
            return int(rng.integers(0, n_actions))
        return int(rng.choice(n_actions, p=p))

    psdp_t0 = time.perf_counter()
    # Backward dynamic programming over timesteps 0..h-2.
    for target_t in range(layer_h - 2, -1, -1):
        for _ in range(psdp_samples):
            env = env_factory()
            obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            obs = np.asarray(obs, dtype=np.int32)
            terminated = truncated = False

            # Prefix rollout to reach x_l.
            for t in range(target_t):
                pol = cover_mixtures[t].sample_policy(rng)
                a_pref = pol.act(obs, t, rng)
                obs, _r, terminated, truncated, _ = env.step(a_pref)
                obs = np.asarray(obs, dtype=np.int32)
                if terminated or truncated:
                    break
            if terminated or truncated:
                continue

            x_l = np.asarray(obs, dtype=np.int32).copy()
            sx = int(x_l[0])
            sy = int(x_l[1])

            # Explore actions at layer l for Q regression.
            a_l = int(rng.integers(0, n_actions))
            obs_next, _r_env, terminated, truncated, _ = env.step(a_l)
            obs_next = np.asarray(obs_next, dtype=np.int32)
            ret = float(w_hat.prob(x_l, a_l, obs_next))
            obs = obs_next

            # Suffix rollout using already-computed future policies.
            # Reward is given only at paper step h (the transition x_{h-1}->x_h),
            # which is timestep (layer_h - 2) in 0-based indexing.
            if target_t == layer_h - 2:
                ret = float(w_hat.prob(x_l, a_l, obs_next))
            for t in range(target_t + 1, layer_h - 1):
                if terminated or truncated:
                    break
                a_suf = _sample_from_future_policy(obs, t)
                obs_next, _r_env, terminated, truncated, _ = env.step(a_suf)
                obs_next = np.asarray(obs_next, dtype=np.int32)
                if t == layer_h - 2:
                    ret = float(w_hat.prob(obs, a_suf, obs_next))
                obs = obs_next

            q_sums[target_t, sx, sy, a_l] += float(ret)
            q_counts[target_t, sx, sy, a_l] += 1

        for x in range(length + 1):
            for y in range(width + 1):
                cnts = q_counts[target_t, x, y]
                if int(np.sum(cnts)) == 0:
                    continue
                means = np.divide(
                    q_sums[target_t, x, y],
                    np.maximum(cnts, 1),
                    dtype=np.float64,
                )
                best_a = int(np.argmax(means))
                p = np.full(n_actions, epsilon / max(n_actions, 1), dtype=np.float64)
                p[best_a] += 1.0 - epsilon
                policy_logits[target_t, x, y] = p

    xs: list[int] = []
    ys: list[int] = []
    hs: list[int] = []
    ys_action: list[int] = []
    sample_weights: list[float] = []
    for t in range(layer_h - 1):
        for x in range(length + 1):
            for y in range(width + 1):
                cnts = q_counts[t, x, y]
                total = int(np.sum(cnts))
                if total == 0:
                    continue
                means = np.divide(q_sums[t, x, y], np.maximum(cnts, 1), dtype=np.float64)
                best_a = int(np.argmax(means))
                xs.append(x)
                ys.append(y)
                hs.append(t)
                ys_action.append(best_a)
                sample_weights.append(float(total))

    if not xs:
        model = CoordPolicy(n_actions=n_actions, hidden_dim=64)
        policy = NeuralPolicy(
            model=model,
            n_actions=n_actions,
            rows=length + 1,
            cols=width + 1,
            horizon=layer_h,
            device_preference=str(device),
        )
        policy.training_stats = {
            "final_loss": None,
            "n_supervised_samples": 0,
            "max_epochs": 0,
        }
        if verbose:
            print(
                f"  [timing] psdp training: {time.perf_counter() - psdp_t0:.3f}s"
            )
        return policy

    x_t = torch.tensor(xs, dtype=torch.float32, device=device)
    y_t = torch.tensor(ys, dtype=torch.float32, device=device)
    h_t = torch.tensor(hs, dtype=torch.float32, device=device)
    target_t = torch.tensor(ys_action, dtype=torch.long, device=device)
    weight_t = torch.tensor(sample_weights, dtype=torch.float32, device=device)

    model = CoordPolicy(n_actions=n_actions, hidden_dim=64).to(device)
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    max_epochs = 200
    final_loss = 0.0
    for _ in range(max_epochs):
        logits = model(
            x_t,
            y_t,
            h_t,
            rows=length + 1,
            cols=width + 1,
            horizon=layer_h,
        )
        loss_per = F.cross_entropy(logits, target_t, reduction="none")
        loss = torch.sum(loss_per * weight_t) / torch.clamp(torch.sum(weight_t), min=1.0)
        final_loss = float(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    policy = NeuralPolicy(
        model=model,
        n_actions=n_actions,
        rows=length + 1,
        cols=width + 1,
        horizon=layer_h,
        device_preference=str(device),
    )
    policy.training_stats = {
        "final_loss": final_loss,
        "n_supervised_samples": int(len(xs)),
        "max_epochs": int(max_epochs),
    }
    if verbose:
        print(f"  [timing] psdp training: {time.perf_counter() - psdp_t0:.3f}s")
    return policy

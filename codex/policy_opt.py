"""Policy optimization subroutine for Algorithm 4 (tabular PSDP)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from numpy.random import Generator

from models.weight_fn import TabularWeightModel

from .rollouts import PolicyMixture, TabularPolicy, state_key, tightrope_predict_next


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
) -> TabularPolicy:
    """
    ``PolicyOptimization_{h-1}(...)`` via tabular PSDP-style dynamic programming.

    For each layer ``l`` (0-based timestep) from ``h-2`` down to ``0``, collect online
    tuples ``(x_l, a_l, R_l)`` where reward uses ``w_hat`` on observed transitions and
    suffix actions follow already-computed future policies. Regress tabular ``Q_l`` by
    sample means, then extract greedy policy with epsilon-greedy smoothing.
    """
    del delta_opt, epsilon_opt

    if layer_h < 2:
        raise ValueError("layer_h must be >= 2")
    if len(cover_mixtures) < layer_h - 1:
        raise ValueError("need p_1..p_{h-1} cover mixtures")

    epsilon = float(np.clip(psdp_epsilon_greedy, 0.0, 1.0))
    timestep_tables: dict[int, dict[Any, np.ndarray]] = {}
    q_tables: dict[int, dict[Any, np.ndarray]] = {}

    def _sample_from_future_policy(obs: np.ndarray, timestep: int) -> int:
        key = state_key(obs)
        p = timestep_tables.get(timestep, {}).get(key)
        if p is None:
            return int(rng.integers(0, n_actions))
        return int(rng.choice(n_actions, p=p))

    # Backward dynamic programming over timesteps 0..h-2.
    for target_t in range(layer_h - 2, -1, -1):
        returns_by_sa: dict[tuple[Any, int], list[float]] = defaultdict(list)

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
            state_l_key = state_key(x_l)

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

            returns_by_sa[(state_l_key, a_l)].append(ret)

        # Tabular regression: Q(s,a) = empirical mean return.
        q_by_state: dict[Any, np.ndarray] = {}
        for (s_key, a), vals in returns_by_sa.items():
            q = q_by_state.get(s_key)
            if q is None:
                q = np.zeros(n_actions, dtype=np.float64)
                q_by_state[s_key] = q
            q[a] = float(np.mean(vals))
        q_tables[target_t] = q_by_state

        pi_l: dict[Any, np.ndarray] = {}
        for s_key, q in q_by_state.items():
            best_a = int(np.argmax(q))
            p = np.full(n_actions, epsilon / max(n_actions, 1), dtype=np.float64)
            p[best_a] += 1.0 - epsilon
            pi_l[s_key] = p
        timestep_tables[target_t] = pi_l

    # Export as time-dependent table keyed by (timestep, state_key).
    probs: dict[Any, np.ndarray] = {}
    for t, pi_t in timestep_tables.items():
        for s_key, p in pi_t.items():
            probs[(int(t), s_key)] = p

    return TabularPolicy(n_actions=n_actions, probs=probs, q_values=q_tables)

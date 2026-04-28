"""Policy optimization subroutine for Algorithm 4 (tabular PSDP)."""

from __future__ import annotations

from collections import defaultdict
import multiprocessing as mp
from typing import Any

import numpy as np
from numpy.random import Generator

from models.weight_fn import TabularWeightModel

from .rollouts import PolicyMixture, TabularPolicy, state_key

_POOL_ENV_FACTORY: Any | None = None


def _init_psdp_pool(env_factory: Any) -> None:
    global _POOL_ENV_FACTORY
    _POOL_ENV_FACTORY = env_factory


def _distance_bonus_value(
    obs: np.ndarray,
    start_obs: np.ndarray,
    *,
    width: int,
    length: int,
    use_distance_bonus: bool,
) -> float:
    if not use_distance_bonus:
        return 0.0
    if int(obs[0]) == int(length) and int(obs[1]) == int(width):
        return 0.0
    dist = abs(int(obs[0]) - int(start_obs[0])) + abs(int(obs[1]) - int(start_obs[1]))
    return 0.01 * float(dist)


def _psdp_rollout_worker(
    args: tuple[
        PolicyMixture,
        dict[int, dict[Any, np.ndarray]],
        TabularWeightModel,
        int,
        int,
        int,
        int,
        int,
        bool,
        list[int],
    ]
) -> dict[tuple[Any, int], tuple[float, int]]:
    (
        cover_mixture_t,
        timestep_tables,
        w_hat,
        target_t,
        layer_h,
        width,
        length,
        n_actions,
        use_distance_bonus,
        seeds_batch,
    ) = args
    if _POOL_ENV_FACTORY is None:
        raise RuntimeError("psdp pool env_factory is not initialized")
    local_sum_cnt: dict[tuple[Any, int], list[float | int]] = {}
    for seed in seeds_batch:
        rng = np.random.default_rng(seed)
        env = _POOL_ENV_FACTORY()
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        obs = np.asarray(obs, dtype=np.int32)
        start_obs = obs.copy()
        terminated = truncated = False

        prefix_policy = cover_mixture_t.sample_policy(rng)
        for t in range(target_t):
            a_pref = prefix_policy.act(obs, t, rng)
            obs, _r, terminated, truncated, _ = env.step(a_pref)
            obs = np.asarray(obs, dtype=np.int32)
            if terminated or truncated:
                break
        if terminated or truncated:
            continue

        x_l = np.asarray(obs, dtype=np.int32).copy()
        state_l_key = state_key(x_l)
        a_l = int(rng.integers(0, n_actions))
        obs_next, _r_env, terminated, truncated, _ = env.step(a_l)
        obs_next = np.asarray(obs_next, dtype=np.int32)
        ret = float(w_hat.prob_state(obs_next)) + _distance_bonus_value(
            obs_next,
            start_obs,
            width=width,
            length=length,
            use_distance_bonus=use_distance_bonus,
        )
        obs = obs_next

        for t in range(target_t + 1, layer_h - 1):
            if terminated or truncated:
                break
            key = state_key(obs)
            p = timestep_tables.get(t, {}).get(key)
            if p is None:
                a_suf = int(rng.integers(0, n_actions))
            else:
                a_suf = int(rng.choice(n_actions, p=p))
            obs_next, _r_env, terminated, truncated, _ = env.step(a_suf)
            obs_next = np.asarray(obs_next, dtype=np.int32)
            ret += float(w_hat.prob_state(obs_next)) + _distance_bonus_value(
                obs_next,
                start_obs,
                width=width,
                length=length,
                use_distance_bonus=use_distance_bonus,
            )
            obs = obs_next

        key_sa = (state_l_key, a_l)
        cur = local_sum_cnt.get(key_sa)
        if cur is None:
            local_sum_cnt[key_sa] = [float(ret), 1]
        else:
            cur[0] = float(cur[0]) + float(ret)
            cur[1] = int(cur[1]) + 1

    return {k: (float(v[0]), int(v[1])) for k, v in local_sum_cnt.items()}


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
    psdp_sample_workers: int = 8,
    epsilon_opt: float = 0.05,
    delta_opt: float = 0.01,
    psdp_epsilon_greedy: float = 0.05,
    use_distance_bonus: bool = True,
) -> TabularPolicy:
    """
    ``PolicyOptimization_{h-1}(...)`` via tabular PSDP-style dynamic programming.

    For each layer ``l`` (0-based timestep) from ``h-2`` down to ``0``, collect online
    tuples ``(x_l, a_l, R_l)`` where reward uses ``w_hat`` on observed transitions and
    suffix actions follow already-computed future policies. Regress tabular ``Q_l`` by
    sample means, then extract greedy policy with epsilon-greedy smoothing.
    """
    del delta_opt, epsilon_opt
    global _POOL_ENV_FACTORY

    if layer_h < 2:
        raise ValueError("layer_h must be >= 2")
    if len(cover_mixtures) < layer_h - 1:
        raise ValueError("need p_1..p_{h-1} cover mixtures")

    epsilon = float(np.clip(psdp_epsilon_greedy, 0.0, 1.0))
    timestep_tables: dict[int, dict[Any, np.ndarray]] = {}
    q_tables: dict[int, dict[Any, np.ndarray]] = {}

    workers = max(int(psdp_sample_workers), 1)
    pool = None
    if workers > 1:
        try:
            pool = mp.get_context("fork").Pool(
                processes=workers, initializer=_init_psdp_pool, initargs=(env_factory,)
            )
        except ValueError:
            pool = mp.get_context().Pool(
                processes=workers, initializer=_init_psdp_pool, initargs=(env_factory,)
            )

    try:
        # Backward dynamic programming over timesteps 0..h-2.
        for target_t in range(layer_h - 2, -1, -1):
            sum_cnt_by_sa: dict[tuple[Any, int], list[float | int]] = {}
            seeds = [
                int(x)
                for x in rng.integers(0, 2**31 - 1, size=psdp_samples, dtype=np.int64)
            ]
            batch_size = max(1, psdp_samples // max(4 * workers, 1))
            seed_batches = [
                seeds[i : i + batch_size] for i in range(0, len(seeds), batch_size)
            ]
            tasks = [
                (
                    cover_mixtures[target_t],
                    timestep_tables,
                    w_hat,
                    target_t,
                    layer_h,
                    width,
                    length,
                    n_actions,
                    use_distance_bonus,
                    seed_batch,
                )
                for seed_batch in seed_batches
            ]
            if pool is None:
                _init_psdp_pool(env_factory)
                results = [_psdp_rollout_worker(task) for task in tasks]
            else:
                results = pool.map(_psdp_rollout_worker, tasks, chunksize=1)

            for local_stats in results:
                for key_sa, (sum_ret, cnt) in local_stats.items():
                    cur = sum_cnt_by_sa.get(key_sa)
                    if cur is None:
                        sum_cnt_by_sa[key_sa] = [float(sum_ret), int(cnt)]
                    else:
                        cur[0] = float(cur[0]) + float(sum_ret)
                        cur[1] = int(cur[1]) + int(cnt)

            # Tabular regression: Q(s,a) = empirical mean return.
            q_by_state: dict[Any, np.ndarray] = {}
            for (s_key, a), (sum_ret, cnt) in sum_cnt_by_sa.items():
                if int(cnt) <= 0:
                    continue
                q = q_by_state.get(s_key)
                if q is None:
                    q = np.zeros(n_actions, dtype=np.float64)
                    q_by_state[s_key] = q
                q[a] = float(sum_ret) / float(cnt)
            q_tables[target_t] = q_by_state

            pi_l: dict[Any, np.ndarray] = {}
            for s_key, q in q_by_state.items():
                best_a = int(np.argmax(q))
                p = np.full(n_actions, epsilon / max(n_actions, 1), dtype=np.float64)
                p[best_a] += 1.0 - epsilon
                pi_l[s_key] = p
            timestep_tables[target_t] = pi_l
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _POOL_ENV_FACTORY = None

    # Export as time-dependent table keyed by (timestep, state_key).
    probs: dict[Any, np.ndarray] = {}
    for t, pi_t in timestep_tables.items():
        for s_key, p in pi_t.items():
            probs[(int(t), s_key)] = p

    return TabularPolicy(n_actions=n_actions, probs=probs, q_values=q_tables)

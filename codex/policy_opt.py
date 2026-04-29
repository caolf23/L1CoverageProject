"""Policy optimization subroutine for Algorithm 4 (tabular PSDP)."""

from __future__ import annotations

from collections import defaultdict
import multiprocessing as mp
from typing import Any

import numpy as np
from numpy.random import Generator
import torch
import torch.nn as nn

from models.weight_fn import TabularWeightModel

from .rollouts import PolicyMixture, TabularPolicy, state_key

_POOL_ENV_FACTORY: Any | None = None
_SINUSOIDAL_N_FREQ = 8
_NN_HIDDEN_DIM = 64
_NN_EPOCHS = 16
_NN_BATCH_SIZE = 64
_NN_LR = 2e-2
_NN_L2 = 1e-4
_NN_GRAD_CLIP = 1.0


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


def _obs_from_state_key(state: Any) -> np.ndarray:
    arr = np.asarray(state, dtype=np.int32).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"state key must have at least 2 entries, got {state}")
    return np.array([int(arr[0]), int(arr[1])], dtype=np.int32)


def _sinusoidal_xy_encoding(
    obs: np.ndarray,
    *,
    width: int,
    length: int,
    n_freq: int = _SINUSOIDAL_N_FREQ,
) -> np.ndarray:
    x = int(obs[0])
    y = int(obs[1])
    if x == int(length) and y == int(width):
        # Reserve a distinct bucket for absorbing state.
        x_norm = 1.0
        y_norm = 1.0
    else:
        x_norm = float(x) / float(max(length - 1, 1))
        y_norm = float(y) / float(max(width - 1, 1))
    vals = np.empty(4 * n_freq, dtype=np.float64)
    for i in range(n_freq):
        omega = float(2**i) * np.pi
        vals[4 * i + 0] = np.sin(omega * x_norm)
        vals[4 * i + 1] = np.cos(omega * x_norm)
        vals[4 * i + 2] = np.sin(omega * y_norm)
        vals[4 * i + 3] = np.cos(omega * y_norm)
    return vals


def _build_weighted_sa_dataset(
    sum_cnt_by_sa: dict[tuple[Any, int], list[float | int]],
    *,
    width: int,
    length: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_rows: list[np.ndarray] = []
    a_rows: list[int] = []
    y_rows: list[float] = []
    w_rows: list[float] = []
    for (s_key, a), (sum_ret, cnt) in sum_cnt_by_sa.items():
        cnt_i = int(cnt)
        if cnt_i <= 0:
            continue
        obs = _obs_from_state_key(s_key)
        x_rows.append(_sinusoidal_xy_encoding(obs, width=width, length=length))
        a_rows.append(int(a))
        y_rows.append(float(sum_ret) / float(cnt_i))
        w_rows.append(float(cnt_i))
    if not x_rows:
        return (
            np.empty((0, 4 * _SINUSOIDAL_N_FREQ), dtype=np.float64),
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=np.float64),
            np.empty((0,), dtype=np.float64),
        )
    return (
        np.asarray(x_rows, dtype=np.float64),
        np.asarray(a_rows, dtype=np.int64),
        np.asarray(y_rows, dtype=np.float64),
        np.asarray(w_rows, dtype=np.float64),
    )


def _train_q_network_weighted(
    x: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    sample_weights: np.ndarray,
    *,
    n_actions: int,
    rng: Generator,
    hidden_dim: int = _NN_HIDDEN_DIM,
    epochs: int = _NN_EPOCHS,
    batch_size: int = _NN_BATCH_SIZE,
    lr: float = _NN_LR,
    l2: float = _NN_L2,
    grad_clip: float = _NN_GRAD_CLIP,
) -> nn.Module:
    seed = int(rng.integers(0, 2**31 - 1))
    torch.manual_seed(seed)
    input_dim = int(x.shape[1])
    model = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, n_actions),
    )
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=l2)

    n = int(x.shape[0])
    if n == 0:
        return model
    bs = max(1, min(int(batch_size), n))
    sw_total = float(np.sum(sample_weights))
    if sw_total <= 0:
        sw_total = 1.0

    x_t = torch.from_numpy(x.astype(np.float32))
    a_t = torch.from_numpy(actions.astype(np.int64))
    y_t = torch.from_numpy(targets.astype(np.float32))
    w_t = torch.from_numpy((sample_weights / sw_total).astype(np.float32))

    for _ in range(max(int(epochs), 1)):
        perm = torch.from_numpy(rng.permutation(n).astype(np.int64))
        for start in range(0, n, bs):
            idx = perm[start : start + bs]
            xb = x_t[idx]
            ab = a_t[idx]
            yb = y_t[idx]
            wb = w_t[idx]
            q = model(xb)
            pred = q.gather(1, ab.unsqueeze(1)).squeeze(1)
            loss = torch.sum(wb * (pred - yb) ** 2)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            optimizer.step()
    return model


def _predict_q_values(x: np.ndarray, model: nn.Module) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32))
        q = model(x_t)
    return q.cpu().numpy().astype(np.float64)


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

            # Fit one NN Q-function for this timestep, then export tabular values on
            # visited states to keep the rest of the pipeline unchanged.
            feats, actions, targets, sample_weights = _build_weighted_sa_dataset(
                sum_cnt_by_sa, width=width, length=length
            )
            q_by_state: dict[Any, np.ndarray] = {}
            if feats.shape[0] > 0:
                nn_params = _train_q_network_weighted(
                    feats,
                    actions,
                    targets,
                    sample_weights,
                    n_actions=n_actions,
                    rng=rng,
                )
                state_keys = list({s_key for (s_key, _a) in sum_cnt_by_sa.keys()})
                x_states = np.asarray(
                    [
                        _sinusoidal_xy_encoding(
                            _obs_from_state_key(s_key), width=width, length=length
                        )
                        for s_key in state_keys
                    ],
                    dtype=np.float64,
                )
                q_pred = _predict_q_values(x_states, nn_params)
                for idx, s_key in enumerate(state_keys):
                    q_by_state[s_key] = np.asarray(q_pred[idx], dtype=np.float64).copy()
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

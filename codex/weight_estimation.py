"""Algorithm 5: weight function estimation."""

from __future__ import annotations

import multiprocessing as mp
from typing import Any

import numpy as np
from numpy.random import Generator

from models.weight_fn import TabularWeightModel, n_weight_samples

from .rollouts import Policy, PolicyMixture, build_q_mixture

_POOL_ENV_FACTORY: Any | None = None


def _sample_mixture_with_retry_worker(
    args: tuple[PolicyMixture, int, int, int]
) -> list[np.ndarray] | None:
    mix, layer_h, seed, outer_attempts = args
    if _POOL_ENV_FACTORY is None:
        raise RuntimeError("pool env_factory is not initialized")
    rng = np.random.default_rng(seed)
    for _ in range(outer_attempts):
        env = _POOL_ENV_FACTORY()
        pol = mix.sample_policy(rng)
        rollout_states = _sample_rollout_states(env, pol, layer_h=layer_h, rng=rng)
        if rollout_states is not None:
            return rollout_states
    return None


def _sample_policy_worker(
    args: tuple[Policy, int, int]
) -> list[np.ndarray] | None:
    policy, layer_h, seed = args
    if _POOL_ENV_FACTORY is None:
        raise RuntimeError("pool env_factory is not initialized")
    rng = np.random.default_rng(seed)
    env = _POOL_ENV_FACTORY()
    return _sample_rollout_states(env, policy, layer_h=layer_h, rng=rng)


def _sample_rollout_states(
    env: Any,
    policy: Policy,
    *,
    layer_h: int,
    rng: Generator,
    max_attempts: int = 512,
) -> list[np.ndarray] | None:
    """
    Sample one rollout prefix ``[x_0, x_1, ..., x_{h-1}]`` for 1-based paper layer ``h``.
    """
    if layer_h < 2:
        raise ValueError("layer_h must be >= 2")
    target_steps = max(layer_h - 1, 1)
    for _ in range(max_attempts):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        obs = np.asarray(obs, dtype=np.int32)
        states = [obs.copy()]
        terminated = truncated = False
        t = 0
        while t < target_steps and not (terminated or truncated):
            a = policy.act(obs, t, rng)
            obs, _r, terminated, truncated, _ = env.step(a)
            obs = np.asarray(obs, dtype=np.int32)
            states.append(obs.copy())
            t += 1
        if len(states) == target_steps + 1:
            return states
    return None


def estimate_weight_function(
    env_factory: Any,
    *,
    layer_h: int,
    iteration_t: int,
    p_h_minus_1: PolicyMixture,
    history_policies: list[Policy],
    epsilon_w: float,
    delta_w: float,
    width: int,
    length: int,
    n_actions: int,
    rng: Generator,
    fit_steps: int = 300,
    n_weight_cap: int | None = 256,
    fit_lr: float = 0.12,
    fit_l2: float = 1e-4,
    fit_lr_decay: float = 0.997,
    fit_patience: int = 60,
    weight_sample_workers: int = 8,
    zero_absorbing_after_fit: bool = False,
) -> tuple[TabularWeightModel, dict[str, float]]:
    """
    ``EstimateWeightFunction_{h,t}(p_{h-1}, {pi^i}_{i<t}; eps, delta, W)``.

    When ``t == 1``, ``q := p_{h-1}``. When ``t >= 2``, use ``build_q_mixture``.
    """
    if layer_h < 2:
        raise ValueError("layer_h must be >= 2")

    model = TabularWeightModel(
        length=length,
        width=width,
        n_actions=n_actions,
        zero_absorbing_after_fit=zero_absorbing_after_fit,
    )
    n = n_weight_samples(epsilon_w, delta_w, model.log_w_class_size)
    if n_weight_cap is not None:
        n = min(n, int(n_weight_cap))
    n = max(n, 4)

    d1: list[np.ndarray] = []
    d2: list[np.ndarray] = []

    def draw_from_mixture(
        mix: PolicyMixture, local_rng: Generator
    ) -> list[np.ndarray] | None:
        env = env_factory()
        pol = mix.sample_policy(local_rng)
        return _sample_rollout_states(env, pol, layer_h=layer_h, rng=local_rng)

    if iteration_t == 1:
        q = p_h_minus_1
    else:
        q = build_q_mixture(
            p_h_minus_1,
            history_policies,
            layer_h=layer_h,
            n_actions=n_actions,
            rng=rng,
        )

    workers = max(int(weight_sample_workers), 1)
    global _POOL_ENV_FACTORY
    _POOL_ENV_FACTORY = env_factory
    pool = None
    if workers > 1:
        try:
            pool = mp.get_context("fork").Pool(processes=workers)
        except ValueError:
            # Fallback for platforms where "fork" context is unavailable.
            pool = mp.get_context().Pool(processes=workers)

    def _collect_mixture_samples(
        mix: PolicyMixture, count: int
    ) -> list[list[np.ndarray]]:
        seeds = [int(x) for x in rng.integers(0, 2**31 - 1, size=count, dtype=np.int64)]
        if pool is None:
            out: list[list[np.ndarray]] = []
            for seed in seeds:
                local_rng = np.random.default_rng(seed)
                tr = None
                for _ in range(256):
                    rollout_states = draw_from_mixture(mix, local_rng)
                    if rollout_states is not None:
                        break
                if rollout_states is not None:
                    out.append(rollout_states)
            return out

        tasks = [(mix, layer_h, seed, 256) for seed in seeds]
        chunk = max(1, count // max(4 * workers, 1))
        results = pool.map(_sample_mixture_with_retry_worker, tasks, chunksize=chunk)
        return [rollout_states for rollout_states in results if rollout_states is not None]

    def _collect_policy_samples(policy: Policy, count: int) -> list[list[np.ndarray]]:
        seeds = [int(x) for x in rng.integers(0, 2**31 - 1, size=count, dtype=np.int64)]
        if pool is None:
            out: list[list[np.ndarray]] = []
            for seed in seeds:
                local_rng = np.random.default_rng(seed)
                env = env_factory()
                rollout_states = _sample_rollout_states(
                    env, policy, layer_h=layer_h, rng=local_rng
                )
                if rollout_states is not None:
                    out.append(rollout_states)
            return out

        tasks = [(policy, layer_h, seed) for seed in seeds]
        chunk = max(1, count // max(4 * workers, 1))
        results = pool.map(_sample_policy_worker, tasks, chunksize=chunk)
        return [rollout_states for rollout_states in results if rollout_states is not None]

    try:
        # Initial n samples (Algorithm 5 line 5): same rollout positions into D1 and D2.
        init_samples = _collect_mixture_samples(q, n)
        for states in init_samples:
            d1.extend(states)
            d2.extend(states)

        # For each historical policy index i in 1..t-1 (0-based: i < t-1)
        for i in range(iteration_t - 1):
            pi_i = history_policies[i]

            q_samples = _collect_mixture_samples(q, n)
            for states in q_samples:
                d1.extend(states)

            pi_samples = _collect_policy_samples(pi_i, n)

            m = min(len(q_samples), len(pi_samples))
            for j in range(m):
                xtilde_states = pi_samples[j]
                d2.extend(xtilde_states)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _POOL_ENV_FACTORY = None

    if not d1:
        return model, {
            "layer": float(layer_h),
            "iteration": float(iteration_t),
            "n_d1": 0.0,
            "n_d2": 0.0,
            "objective_before": 0.0,
            "objective_after": 0.0,
        }

    fit_stats = model.fit(
        d1,
        d2,
        iteration_t,
        steps=fit_steps,
        lr=fit_lr,
        l2=fit_l2,
        lr_decay=fit_lr_decay,
        patience=fit_patience,
    )
    metrics = {
        "layer": float(layer_h),
        "iteration": float(iteration_t),
        "n_d1": float(len(d1)),
        "n_d2": float(len(d2)),
        "objective_before": float(fit_stats["objective_before"]),
        "objective_after": float(fit_stats["objective_after"]),
    }
    return model, metrics

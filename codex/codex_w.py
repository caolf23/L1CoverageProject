"""Algorithm 4: CODEX.W policy cover construction."""

from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
from numpy.random import Generator

from .policy_opt import policy_optimization_h_minus_1
from .rollouts import (
    ComposedUniformPolicy,
    PolicyMixture,
    UniformRandomPolicy,
)
from .weight_estimation import estimate_weight_function


def _dump_w_model(
    w_model: Any,
    *,
    width: int,
    length: int,
    n_actions: int,
) -> None:
    print("  w_model:")
    if hasattr(w_model, "prob_state"):
        n_states = length * width
        dead_obs = np.array([length, width], dtype=np.int32)
        start_obs = np.array([0, width // 2], dtype=np.int32)

        def _distance_bonus(obs: np.ndarray) -> float:
            if int(obs[0]) == int(length) and int(obs[1]) == int(width):
                return 0.0
            dist = abs(int(obs[0]) - int(start_obs[0])) + abs(int(obs[1]) - int(start_obs[1]))
            return 0.01 * float(dist)

        for s_idx in range(n_states):
            x = s_idx // width
            y = s_idx % width
            obs = np.array([x, y], dtype=np.int32)
            w_val = float(w_model.prob_state(obs))
            d_val = _distance_bonus(obs)
            total = w_val + d_val
            print(
                f"    state:{s_idx},state_reward:{w_val:.6f},"
                f"distance_bonus:{d_val:.6f},total_reward:{total:.6f}"
            )

        # Also show state reward on absorbing/dead state.
        dead_idx = n_states
        w_val = float(w_model.prob_state(dead_obs))
        d_val = _distance_bonus(dead_obs)
        total = w_val + d_val
        print(
            f"    state:{dead_idx},state_reward:{w_val:.6f},"
            f"distance_bonus:{d_val:.6f},total_reward:{total:.6f}"
        )
    else:
        print("    object=", w_model)


def _dump_policy(policy: Any, policy_name: str = "pi_new") -> None:
    print(f"  {policy_name}:")
    if hasattr(policy, "probs"):
        probs = getattr(policy, "probs")
        state_keys = set()
        for k in probs.keys():
            if (
                isinstance(k, tuple)
                and len(k) == 2
                and isinstance(k[0], int)
                and isinstance(k[1], tuple)
            ):
                state_keys.add(k[1])
            else:
                state_keys.add(k)
        print("    n_states_with_probs=", len(state_keys))
        print("    n_table_entries=", len(probs))
        print("    probs(full)=", probs)
        q_values = getattr(policy, "q_values", None)
        if isinstance(q_values, dict):
            print("    q_values:")
            for t in sorted(q_values.keys()):
                q_t = q_values[t]
                print(f"      timestep={t}, n_states={len(q_t)}")
                print(f"      table={q_t}")
    else:
        print("    object=", policy)


def _dump_stitched_cover(h: int, cover: PolicyMixture) -> None:
    print(f"  stitched cover p_{h}:")
    print("    n_components=", len(cover.policies))
    print("    weights=", np.array2string(cover.weights, precision=4, threshold=np.inf))
    for idx, pol in enumerate(cover.policies):
        first_uniform = getattr(pol, "first_uniform_timestep", None)
        base = getattr(pol, "base", None)
        print(
            f"    component[{idx}]: type={type(pol).__name__}, "
            f"first_uniform_timestep={first_uniform}, "
            f"base_type={type(base).__name__ if base is not None else 'N/A'}"
        )


def run_codex_w(
    env_factory: Any,
    *,
    horizon_h: int,
    epsilon: float,
    delta: float,
    n_actions: int,
    width: int,
    length: int,
    rng: Generator | None = None,
    c_push: float = 1.0,
    c_const: float = 0.1,
    c_prime: float = 0.1,
    epsilon_w: float | None = None,
    weight_fit_steps: int = 700,
    psdp_samples: int = 600,
    n_weight_cap: int | None = 96,
    weight_sample_workers: int = 8,
    weight_fit_lr: float = 0.12,
    weight_fit_l2: float = 0,
    weight_fit_lr_decay: float = 0.997,
    weight_fit_patience: int = 60,
    weight_zero_absorbing_after_fit: bool = False,
    psdp_epsilon_greedy: float = 0.05,
    on_layer_complete: Callable[[int, PolicyMixture], None] | None = None,
    verbose: bool = False,
    return_diagnostics: bool = False,
) -> (
    tuple[dict[int, PolicyMixture], dict[int, list]]
    | tuple[dict[int, PolicyMixture], dict[int, list], list[dict[str, float]]]
):
    """
    Build policy covers ``(p_1, ..., p_H)`` (1-indexed keys in ``covers`` dict).

    Returns ``(covers, policies)`` where ``policies[h]`` lists ``π^{h,1},...,π^{h,T}``.
    If ``return_diagnostics`` is True, appends per-(h,t) Algorithm 5 fit metrics.
    """
    if horizon_h < 2:
        raise ValueError("horizon_h must be at least 2")
    rng = rng or np.random.default_rng()

    t_rounds = max(int(math.ceil(1.0 / max(epsilon, 1e-9))), 1)
    epsilon_w_eff = epsilon_w
    if epsilon_w_eff is None:
        epsilon_w_eff = float(
            c_const * math.sqrt(max(c_push, 1e-9) / max(n_actions, 1)) * math.sqrt(epsilon)
        )
    epsilon_opt = float(c_prime * epsilon * epsilon)
    delta_w = delta / max(2 * horizon_h * t_rounds, 1)
    delta_opt = delta_w

    covers: dict[int, PolicyMixture] = {
        1: PolicyMixture(
            policies=[UniformRandomPolicy(n_actions=n_actions)],
            weights=np.array([1.0], dtype=np.float64),
        )
    }
    policies: dict[int, list] = {h: [] for h in range(2, horizon_h + 1)}
    diagnostics: list[dict[str, float]] = []

    for h in range(2, horizon_h + 1):
        pis_h: list = []
        for t in range(1, t_rounds + 1):
            # if verbose:
            print("Current h: ", h, "Current t: ", t)
            hist = pis_h[: t - 1]
            w_model, fit_metrics = estimate_weight_function(
                env_factory,
                layer_h=h,
                iteration_t=t,
                p_h_minus_1=covers[h - 1],
                history_policies=hist,
                epsilon_w=epsilon_w_eff,
                delta_w=delta_w,
                width=width,
                length=length,
                n_actions=n_actions,
                rng=rng,
                fit_steps=weight_fit_steps,
                n_weight_cap=n_weight_cap,
                weight_sample_workers=weight_sample_workers,
                fit_lr=weight_fit_lr,
                fit_l2=weight_fit_l2,
                fit_lr_decay=weight_fit_lr_decay,
                fit_patience=weight_fit_patience,
                zero_absorbing_after_fit=weight_zero_absorbing_after_fit,
            )
            if verbose:
                print(
                    "  fit_metrics:",
                    {
                        "n_d1": int(fit_metrics["n_d1"]),
                        "n_d2": int(fit_metrics["n_d2"]),
                        "objective_before": float(fit_metrics["objective_before"]),
                        "objective_after": float(fit_metrics["objective_after"]),
                    },
                )
                _dump_w_model(
                    w_model,
                    width=width,
                    length=length,
                    n_actions=n_actions,
                )
            diagnostics.append(fit_metrics)
            cover_prefix = [covers[i] for i in range(1, h)]
            pi_new = policy_optimization_h_minus_1(
                env_factory,
                cover_prefix,
                w_model,
                layer_h=h,
                width=width,
                length=length,
                n_actions=n_actions,
                rng=rng,
                psdp_samples=psdp_samples,
                epsilon_opt=epsilon_opt,
                delta_opt=delta_opt,
                psdp_epsilon_greedy=psdp_epsilon_greedy,
            )
            if verbose:
                _dump_policy(pi_new, policy_name=f"pi^(h={h},t={t})")
            pis_h.append(pi_new)

        policies[h] = pis_h
        first_u = max(h - 1, 0)
        composed = [
            ComposedUniformPolicy(pi, n_actions, first_u) for pi in pis_h
        ]
        covers[h] = PolicyMixture(
            policies=composed,
            weights=np.full(len(composed), 1.0 / len(composed), dtype=np.float64),
        )
        if verbose:
            _dump_stitched_cover(h, covers[h])
        if on_layer_complete is not None:
            on_layer_complete(h, covers[h])

    if return_diagnostics:
        return covers, policies, diagnostics
    return covers, policies

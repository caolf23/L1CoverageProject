"""
Rollouts and policy composition for layered CODEX sampling.

Semantics (documented engineering choice; verify against paper):
- ``pi o_{(k)} pi_unif``: follow ``base`` for timesteps ``t < first_uniform_timestep``,
  then draw actions uniformly. For mixture ``q`` in Algorithm 5 use
  ``first_uniform_timestep = max(h - 2, 0)`` (paper ``o_{h-1}``). For ``p_h`` in
  Algorithm 4 use ``first_uniform_timestep = max(h - 1, 0)`` (paper ``o_h``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.random import Generator


class Policy(Protocol):
    """Markov policy: action may depend on time index ``timestep`` (0-based)."""

    def act(self, obs: np.ndarray, timestep: int, rng: Generator) -> int:
        ...


@dataclass
class UniformRandomPolicy:
    """Independent uniform actions (stationary)."""

    n_actions: int

    def act(self, obs: np.ndarray, timestep: int, rng: Generator) -> int:
        return int(rng.integers(0, self.n_actions))


@dataclass
class ComposedUniformPolicy:
    """
    Follow ``base`` until ``timestep >= first_uniform_timestep``, then uniform.
    """

    base: Policy
    n_actions: int
    first_uniform_timestep: int

    def act(self, obs: np.ndarray, timestep: int, rng: Generator) -> int:
        if timestep < self.first_uniform_timestep:
            return self.base.act(obs, timestep, rng)
        return int(rng.integers(0, self.n_actions))


@dataclass
class TabularPolicy:
    """
    Tabular stochastic policy.

    Supports both:
    - stationary mapping: ``probs[state_key] -> action probs``
    - time-dependent mapping: ``probs[(timestep, state_key)] -> action probs``
    """

    n_actions: int
    probs: dict[Any, np.ndarray]
    q_values: dict[int, dict[Any, np.ndarray]] | None = None

    def act(self, obs: np.ndarray, timestep: int, rng: Generator) -> int:
        key = state_key(obs)
        p = self.probs.get((int(timestep), key))
        if p is None:
            p = self.probs.get(key)
        if p is None:
            return int(rng.integers(0, self.n_actions))
        return int(rng.choice(self.n_actions, p=p))


@dataclass
class PolicyMixture:
    """Finite mixture ``sum_k w_k * delta_{pi_k}`` (``weights`` sum to 1)."""

    policies: list[Policy]
    weights: np.ndarray

    def __post_init__(self) -> None:
        w = np.asarray(self.weights, dtype=np.float64)
        if len(w) != len(self.policies):
            raise ValueError("weights length must match policies")
        s = float(w.sum())
        if s <= 0:
            raise ValueError("weights must sum to a positive value")
        self.weights = w / s

    def sample_policy(self, rng: Generator) -> Policy:
        idx = int(rng.choice(len(self.policies), p=self.weights))
        return self.policies[idx]


def state_key(obs: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in np.asarray(obs).reshape(-1))


def tightrope_predict_next(
    obs: np.ndarray, action: int, width: int, length: int
) -> tuple[np.ndarray, bool, bool, float]:
    """
    Deterministic one-step model matching ``TightropeEnv.step``.

    Importantly, lava transitions enter the absorbing state ``(length, width)``
    and remain there.
    """
    x, y = int(obs[0]), int(obs[1])
    absorbing = np.array([length, width], dtype=np.int32)
    if x == length and y == width:
        return absorbing, False, False, 0.0
    dx, dy = 0, 0
    if action == 0:
        dy = -1
    elif action == 1:
        dy = 1
    elif action == 2:
        dx = -1
    elif action == 3:
        dx = 1
    else:
        raise ValueError(f"invalid action {action}")

    nx = x + dx
    ny = y + dy

    if ny < 0 or ny >= width:
        return absorbing, False, False, 0.0

    nx = max(0, min(length - 1, nx))
    nobs = np.array([nx, ny], dtype=np.int32)
    if nx == length - 1:
        return nobs, False, False, 1.0
    return nobs, False, False, 0.0


def sample_transition_at_layer(
    env: Any,
    policy: Policy,
    *,
    layer_h: int,
    rng: Generator,
    max_attempts: int = 512,
) -> tuple[np.ndarray, int, np.ndarray] | None:
    """
    One sample ``(x_{h-1}, a_{h-1}, x_h)`` with **1-based** paper layer ``h >= 2``.

    Procedure: reset, take ``h-2`` actions under ``policy`` (timesteps ``0..h-3``),
    then at ``x_{h-1}`` take action ``a_{h-1}`` and observe ``x_h``.
    """
    if layer_h < 2:
        raise ValueError("layer_h must be >= 2")

    for _ in range(max_attempts):
        obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        terminated = truncated = False
        t = 0
        # Need ``h-2`` transitions to reach ``x_{h-1}`` from ``x_1``.
        while t < layer_h - 2 and not (terminated or truncated):
            a = policy.act(np.asarray(obs), t, rng)
            obs, _r, terminated, truncated, _ = env.step(a)
            t += 1
        if terminated or truncated:
            continue

        x_prev = np.asarray(obs, dtype=np.int32).copy()
        a_tm1 = policy.act(np.asarray(obs), layer_h - 2, rng)
        obs2, _r2, _term2, _trunc2, _ = env.step(a_tm1)
        # Keep terminal/truncated transitions instead of rejection-sampling them.
        # This avoids conditioning D1/D2 on "survivable only" trajectories.
        return x_prev, int(a_tm1), np.asarray(obs2, dtype=np.int32)

    return None


def build_q_mixture(
    p_prev: PolicyMixture,
    history_pis: list[Policy],
    *,
    layer_h: int,
    n_actions: int,
    rng: Generator,
) -> PolicyMixture:
    """
    Algorithm 5 mixture ``q`` for ``t >= 2``:

    ``q = 1/2 * p_{h-1} + 1/(2(t-1)) * sum_{i<t} (pi^i o_{h-1} pi_unif)``.
    """
    del rng  # API symmetry with callers that pass a Generator; mixture is deterministic.
    t = len(history_pis) + 1
    if t < 2:
        raise ValueError("build_q_mixture expects t >= 2 (use p_{h-1} alone for t==1)")

    first_u = max(layer_h - 2, 0)
    composed = [
        ComposedUniformPolicy(pi, n_actions, first_u)
        for pi in history_pis
    ]
    sub_weights = np.full(len(composed), 1.0 / (2.0 * (t - 1)), dtype=np.float64)
    policies = list(p_prev.policies) + composed
    weights = np.concatenate([0.5 * p_prev.weights, sub_weights])
    return PolicyMixture(policies=policies, weights=weights)

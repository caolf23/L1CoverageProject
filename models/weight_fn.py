"""Tabular state weight ``w(x)`` for Algorithm 5 (discrete states)."""

from __future__ import annotations

import math
import numpy as np


class TabularWeightModel:
    """
    Tabular parameterization:
    ``w(s) = w_table[s]``.

    States are indexed ``s_idx = x * width + y`` for grid observations ``(x, y)``.
    The absorbing state is represented by coordinates ``(length, width)`` and
    mapped to index ``length * width``.
    """

    def __init__(
        self,
        length: int,
        width: int,
        n_actions: int = 4,
        *,
        zero_absorbing_after_fit: bool = False,
    ) -> None:
        self.length = length
        self.width = width
        self.n_actions = n_actions
        self.n_grid_states = length * width
        self.n_states = self.n_grid_states + 1
        self.zero_absorbing_after_fit = bool(zero_absorbing_after_fit)
        self.eps = 1e-9
        self.w_table = np.full((self.n_states,), self.eps, dtype=np.float64)

    def state_index(self, obs: np.ndarray) -> int:
        x, y = int(obs[0]), int(obs[1])
        if x == self.length and y == self.width:
            return self.n_grid_states
        return x * self.width + y

    def prob_state(self, obs: np.ndarray) -> float:
        si = self.state_index(obs)
        return float(self.w_table[si])

    def log_prob_state(self, obs: np.ndarray) -> float:
        p = np.clip(self.prob_state(obs), self.eps, 1.0 - self.eps)
        return float(np.log(p))

    def _to_index_array(self, data: list[np.ndarray]) -> np.ndarray:
        if not data:
            return np.empty((0,), dtype=np.int64)
        return np.fromiter((self.state_index(obs) for obs in data), dtype=np.int64)

    @property
    def log_w_class_size(self) -> float:
        """Rough ``log |W|`` for sample-size heuristic."""
        n = self.n_states
        return math.log(max(n, 2))

    def fit(
        self,
        d1: list[np.ndarray],
        d2: list[np.ndarray],
        t: int,
        *,
        steps: int = 700,
        lr: float = 0.12,
        l2: float = 1e-4,
        lr_decay: float = 0.997,
        patience: int = 60,
        min_improvement: float = 1e-5,
    ) -> dict[str, float]:
        """
        Closed-form maximization on ``mean_{D1} log w(s) - t * mean_{D2} w(s)``.

        For each state ``s`` with counts ``c1`` in D1 and ``c2`` in D2:
            maximize ``(c1/|D1|) log w - (t*c2/|D2|) w`` over ``w in [eps, 1-eps]``.
        """
        if not d1:
            return {"objective_before": 0.0, "objective_after": 0.0}

        n1 = len(d1)
        n2_raw = len(d2)
        n2 = max(n2_raw, 1)

        d1_si = self._to_index_array(d1)
        d2_si = self._to_index_array(d2)

        objective_before = self.objective(d1, d2, t)

        # Unused with closed-form solver; kept in signature for caller compatibility.
        del steps, lr, l2, lr_decay, patience, min_improvement

        count1 = np.zeros_like(self.w_table, dtype=np.float64)
        count2 = np.zeros_like(self.w_table, dtype=np.float64)
        np.add.at(count1, d1_si, 1.0)
        if d2_si.size > 0:
            np.add.at(count2, d2_si, 1.0)

        c1_pos = count1 > 0.0
        c2_pos = count2 > 0.0
        both_pos = c1_pos & c2_pos

        new_w = np.full_like(self.w_table, self.eps, dtype=np.float64)
        new_w[c1_pos & ~c2_pos] = 1.0 - self.eps

        if np.any(both_pos):
            ratio = np.empty_like(self.w_table, dtype=np.float64)
            np.divide(
                count1 * float(n2),
                float(max(int(t), 1)) * count2 * float(n1),
                out=ratio,
                where=both_pos,
            )
            new_w[both_pos] = ratio[both_pos]

        self.w_table = np.clip(new_w, self.eps, 1.0 - self.eps)
        if self.zero_absorbing_after_fit:
            # Post-fit projection: force all transitions from absorbing source to eps.
            self.w_table[self.n_grid_states] = self.eps
        objective_after = self.objective(d1, d2, t)
        return {
            "objective_before": float(objective_before),
            "objective_after": float(objective_after),
        }

    def objective(
        self,
        d1: list[np.ndarray],
        d2: list[np.ndarray],
        t: int,
    ) -> float:
        """Algorithm 5 objective: ``mean(log w(s) on D1) - t * mean(w(s) on D2)``."""
        if not d1:
            return 0.0
        d1_si = self._to_index_array(d1)
        d1_w = np.clip(self.w_table[d1_si], self.eps, 1.0 - self.eps)
        term1 = float(np.mean(np.log(d1_w)))
        if d2:
            d2_si = self._to_index_array(d2)
            term2 = float(np.mean(self.w_table[d2_si]))
        else:
            term2 = 0.0
        return term1 - float(t) * term2


def n_weight_samples(epsilon: float, delta: float, log_class_size: float) -> int:
    """``n_weight(eps,delta)`` from Algorithm 5 (tabular ``|W|``)."""
    num = 40.0 * (log_class_size + max(math.log(1.0 / delta), 1e-6))
    den = max(epsilon * epsilon, 1e-12)
    return max(int(math.ceil(num / den)), 8)

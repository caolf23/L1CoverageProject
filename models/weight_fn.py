"""Tabular conditional weight ``w(x' | x, a)`` for Algorithm 5 (discrete states)."""

from __future__ import annotations

import math
import numpy as np


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    x_arr = np.asarray(x, dtype=np.float64)
    # Stable sigmoid for both positive/negative values.
    pos = x_arr >= 0.0
    out = np.empty_like(x_arr, dtype=np.float64)
    out[pos] = 1.0 / (1.0 + np.exp(-x_arr[pos]))
    exp_x = np.exp(x_arr[~pos])
    out[~pos] = exp_x / (1.0 + exp_x)
    return out


class TabularWeightModel:
    """
    Pointwise ratio-style parameterization:
    ``w(s'|s,a) = sigmoid(logit[s,a,s'])``.

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
        # small random init breaks symmetry
        rng = np.random.default_rng(0)
        self.logits = 0.01 * rng.standard_normal(
            (self.n_states, n_actions, self.n_states)
        )

    def state_index(self, obs: np.ndarray) -> int:
        x, y = int(obs[0]), int(obs[1])
        if x == self.length and y == self.width:
            return self.n_grid_states
        return x * self.width + y

    def prob(self, obs: np.ndarray, a: int, obs_next: np.ndarray) -> float:
        si = self.state_index(obs)
        sj = self.state_index(obs_next)
        aa = int(a)
        return float(_sigmoid(self.logits[si, aa, sj]))

    def log_prob(self, obs: np.ndarray, a: int, obs_next: np.ndarray) -> float:
        p = np.clip(self.prob(obs, a, obs_next), 1e-12, 1.0)
        return float(np.log(p))

    @property
    def log_w_class_size(self) -> float:
        """Rough ``log |W|`` for sample-size heuristic."""
        n = self.n_states * self.n_actions * self.n_states
        return math.log(max(n, 2))

    def fit(
        self,
        d1: list[tuple[np.ndarray, int, np.ndarray]],
        d2: list[tuple[np.ndarray, int, np.ndarray]],
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
        Gradient ascent on ``mean_{D1} log w - t * mean_{D2} w`` (Algorithm 5).
        """
        if not d1:
            return {"objective_before": 0.0, "objective_after": 0.0}

        n1 = len(d1)
        n2 = max(len(d2), 1)

        def _to_index_arrays(
            data: list[tuple[np.ndarray, int, np.ndarray]],
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            if not data:
                empty = np.empty((0,), dtype=np.int64)
                return empty, empty, empty
            si = np.fromiter((self.state_index(obs) for obs, _, _ in data), dtype=np.int64)
            aa = np.fromiter((int(a) for _, a, _ in data), dtype=np.int64)
            sj = np.fromiter((self.state_index(sp) for _, _, sp in data), dtype=np.int64)
            return si, aa, sj

        d1_si, d1_aa, d1_sj = _to_index_arrays(d1)
        d2_si, d2_aa, d2_sj = _to_index_arrays(d2)

        if d2_si.size > 0:
            all_tr = np.concatenate(
                [
                    np.stack([d1_si, d1_aa, d1_sj], axis=1),
                    np.stack([d2_si, d2_aa, d2_sj], axis=1),
                ],
                axis=0,
            )
        else:
            all_tr = np.stack([d1_si, d1_aa, d1_sj], axis=1)
        transition_scale = float(max(np.unique(all_tr, axis=0).shape[0], 1))

        # print(f"transition_scale: {transition_scale}")
        
        objective_before = self.objective(d1, d2, t)
        best_obj = objective_before
        best_logits = self.logits.copy()
        no_improve = 0
        lr_now = float(lr)

        for _ in range(steps):
            grad = np.zeros_like(self.logits, dtype=np.float64)
            w1 = _sigmoid(self.logits[d1_si, d1_aa, d1_sj])
            np.add.at(grad, (d1_si, d1_aa, d1_sj), (1.0 - w1) / n1)

            if d2_si.size > 0:
                w2 = _sigmoid(self.logits[d2_si, d2_aa, d2_sj])
                np.add.at(
                    grad,
                    (d2_si, d2_aa, d2_sj),
                    -(float(t) / n2) * w2 * (1.0 - w2),
                )
            grad *= transition_scale

            if l2 > 0:
                grad -= l2 * self.logits

            grad_norm = float(np.linalg.norm(grad))
            if grad_norm > 10.0:
                grad *= 10.0 / grad_norm

            self.logits += lr_now * grad
            lr_now *= lr_decay

            obj = self.objective(d1, d2, t)
            if obj > best_obj + min_improvement:
                best_obj = obj
                best_logits = self.logits.copy()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

        self.logits = best_logits
        if self.zero_absorbing_after_fit:
            # Post-fit projection: force all transitions from absorbing source to ~0.
            self.logits[self.n_grid_states, :, :] = -40.0
        objective_after = best_obj
        return {
            "objective_before": float(objective_before),
            "objective_after": float(objective_after),
        }

    def objective(
        self,
        d1: list[tuple[np.ndarray, int, np.ndarray]],
        d2: list[tuple[np.ndarray, int, np.ndarray]],
        t: int,
    ) -> float:
        """Algorithm 5 objective: ``mean(log w on D1) - t * mean(w on D2)``."""
        if not d1:
            return 0.0
        term1 = float(np.mean([self.log_prob(x, a, xp) for x, a, xp in d1]))
        if d2:
            term2 = float(np.mean([self.prob(x, a, xp) for x, a, xp in d2]))
        else:
            term2 = 0.0
        return term1 - float(t) * term2


def n_weight_samples(epsilon: float, delta: float, log_class_size: float) -> int:
    """``n_weight(eps,delta)`` from Algorithm 5 (tabular ``|W|``)."""
    num = 40.0 * (log_class_size + max(math.log(1.0 / delta), 1e-6))
    den = max(epsilon * epsilon, 1e-12)
    return max(int(math.ceil(num / den)), 8)

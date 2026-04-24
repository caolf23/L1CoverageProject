"""Neural policy for coordinate-based variable-size environments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from numpy.random import Generator

from .rollouts import TabularPolicy


def resolve_torch_device(device_preference: str = "auto") -> torch.device:
    """Resolve runtime device from ``auto|cpu|cuda`` preference."""
    pref = str(device_preference).strip().lower()
    if pref == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if pref == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


class CoordPolicy(nn.Module):
    """MLP over normalized (x, y, h) coordinates."""

    def __init__(self, n_actions: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.n_actions = int(n_actions)
        self.hidden_dim = int(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_actions),
        )

    @staticmethod
    def _normalize_inputs(
        x: torch.Tensor,
        y: torch.Tensor,
        h: torch.Tensor,
        *,
        rows: int,
        cols: int,
        horizon: int,
    ) -> torch.Tensor:
        # Clamp denominators so width/height/horizon == 1 remains well-defined.
        den_x = float(max(rows - 1, 1))
        den_y = float(max(cols - 1, 1))
        den_h = float(max(horizon - 1, 1))
        x_norm = x.float() / den_x
        y_norm = y.float() / den_y
        h_norm = h.float() / den_h
        return torch.stack([x_norm, y_norm, h_norm], dim=-1)

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        h: torch.Tensor,
        *,
        rows: int,
        cols: int,
        horizon: int,
    ) -> torch.Tensor:
        inp = self._normalize_inputs(x, y, h, rows=rows, cols=cols, horizon=horizon)
        return self.net(inp)


@dataclass
class NeuralPolicy:
    """
    Runtime wrapper exposing the project policy interface:
    ``act(obs, timestep, rng) -> int``.
    """

    model: CoordPolicy
    n_actions: int
    rows: int
    cols: int
    horizon: int
    device_preference: str = "auto"

    def __post_init__(self) -> None:
        self.device = resolve_torch_device(self.device_preference)
        self.model = self.model.to(self.device)
        self.model.eval()

    def act(self, obs: np.ndarray, timestep: int, rng: Generator) -> int:
        x = int(np.asarray(obs)[0])
        y = int(np.asarray(obs)[1])
        with torch.no_grad():
            logits = self.model.forward(
                x=torch.tensor([x], dtype=torch.float32, device=self.device),
                y=torch.tensor([y], dtype=torch.float32, device=self.device),
                h=torch.tensor([int(timestep)], dtype=torch.float32, device=self.device),
                rows=self.rows,
                cols=self.cols,
                horizon=self.horizon,
            )[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        probs = np.asarray(probs, dtype=np.float64)
        probs = probs / np.clip(probs.sum(), 1e-12, np.inf)
        return int(rng.choice(self.n_actions, p=probs))

    def save(self, path: str | Path) -> None:
        payload = {
            "state_dict": self.model.state_dict(),
            "n_actions": int(self.n_actions),
            "rows": int(self.rows),
            "cols": int(self.cols),
            "horizon": int(self.horizon),
            "hidden_dim": int(self.model.hidden_dim),
            "device": str(self.device),
        }
        torch.save(payload, str(path))

    def metadata(self) -> dict[str, int | str]:
        return {
            "policy_type": "CoordPolicy",
            "n_actions": int(self.n_actions),
            "rows": int(self.rows),
            "cols": int(self.cols),
            "horizon": int(self.horizon),
            "hidden_dim": int(self.model.hidden_dim),
            "device": str(self.device),
        }

    def to_tabular_policy(self) -> TabularPolicy:
        """
        Materialize ``pi(a|s,t)`` for all ``t in [0, horizon-1]`` and grid states.
        """
        probs: dict[tuple[int, tuple[int, int]], np.ndarray] = {}
        self.model.eval()
        with torch.no_grad():
            for t in range(self.horizon):
                for x in range(self.rows):
                    for y in range(self.cols):
                        logits = self.model.forward(
                            x=torch.tensor([x], dtype=torch.float32, device=self.device),
                            y=torch.tensor([y], dtype=torch.float32, device=self.device),
                            h=torch.tensor([t], dtype=torch.float32, device=self.device),
                            rows=self.rows,
                            cols=self.cols,
                            horizon=self.horizon,
                        )[0]
                        p = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                        p = np.asarray(p, dtype=np.float64)
                        p = p / np.clip(np.sum(p), 1e-12, np.inf)
                        probs[(int(t), (int(x), int(y)))] = p
        tabular = TabularPolicy(n_actions=int(self.n_actions), probs=probs)
        setattr(
            tabular,
            "_tabularized_from",
            {
                "policy_type": "NeuralPolicy",
                "source_device": str(self.device),
                "horizon": int(self.horizon),
                "rows": int(self.rows),
                "cols": int(self.cols),
            },
        )
        return tabular

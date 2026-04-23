"""Tightrope gridworld: narrow corridor over lava (Gymnasium)."""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class TightropeEnv(gym.Env):
    """
    A 5×100 walkable hallway with lava flanking the two sides along the width.

    The agent moves on integer cells ``(x, y)`` with ``x ∈ [0, length-1]`` (along
    the corridor) and ``y ∈ [0, width-1]`` (across the corridor). Failed moves into
    lava transition into a dedicated absorbing state ``(length, width)`` with reward
    0, and the process stays there thereafter. Reaching ``x == length - 1`` (the far
    end) yields reward 1 but the episode does not terminate; rollout continues until
    ``max_steps`` (truncation) or a lava transition.

    Actions are discrete: 0 up (−y), 1 down (+y), 2 left (−x), 3 right (+x).
    """

    metadata = {"render_modes": ["ansi", "human"]}

    def __init__(
        self,
        width: int = 5,
        length: int = 100,
        max_steps: int | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        if width < 1 or length < 2:
            raise ValueError("width must be >= 1 and length must be >= 2")

        self.width = width
        self.length = length
        self.absorbing_state = np.array([length, width], dtype=np.int32)
        self._max_steps = max_steps if max_steps is not None else 4 * length * width

        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=np.array([0, 0], dtype=np.int32),
            high=np.array([length, width], dtype=np.int32),
            dtype=np.int32,
        )

        self.render_mode = render_mode
        self._pos = np.zeros(2, dtype=np.int32)
        self._steps = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._steps = 0
        start_y = self.width // 2
        self._pos = np.array([0, start_y], dtype=np.int32)
        return self._pos.copy(), {}

    def step(
        self, action: SupportsFloat | np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        a = int(action)
        if a not in (0, 1, 2, 3):
            raise ValueError(f"action must be in 0..3, got {a}")

        dx, dy = 0, 0
        if a == 0:
            dy = -1
        elif a == 1:
            dy = 1
        elif a == 2:
            dx = -1
        elif a == 3:
            dx = 1

        nx = int(self._pos[0]) + dx
        ny = int(self._pos[1]) + dy

        self._steps += 1
        truncated = self._steps >= self._max_steps

        if np.array_equal(self._pos, self.absorbing_state):
            return self._pos.copy(), 0.0, False, truncated, {}

        # Lava: leave the safe strip across the corridor width
        if ny < 0 or ny >= self.width:
            self._pos = self.absorbing_state.copy()
            return self._pos.copy(), 0.0, False, truncated, {}

        nx = max(0, min(self.length - 1, nx))
        self._pos[0] = nx
        self._pos[1] = ny

        if nx == self.length - 1:
            return self._pos.copy(), 1.0, False, truncated, {}

        return self._pos.copy(), 0.0, False, truncated, {}

    def render(self) -> str | None:
        rows = []
        for y in range(-1, self.width + 1):
            line_chars = []
            for x in range(self.length):
                if y < 0 or y >= self.width:
                    ch = "~"
                elif int(self._pos[0]) == x and int(self._pos[1]) == y:
                    ch = "A"
                elif x == self.length - 1:
                    ch = "G"
                else:
                    ch = "."
                line_chars.append(ch)
            rows.append("".join(line_chars))
        text = "\n".join(rows)
        if self.render_mode == "human":
            print(text)
            return None
        return text

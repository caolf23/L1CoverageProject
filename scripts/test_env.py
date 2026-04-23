#!/usr/bin/env python3
"""Smoke-test Gymnasium TightropeEnv and core dependencies (run from any cwd)."""

from __future__ import annotations

import sys
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


_ACTION_NAMES = ("up", "down", "left", "right")


def _print_random_rollout(*, seed: int, width: int, length: int, max_steps: int) -> None:
    """Run one episode with a random policy; ``human`` render prints the grid each step."""
    import numpy as np

    from environment import TightropeEnv

    rng = np.random.default_rng(seed)
    env = TightropeEnv(
        width=width,
        length=length,
        max_steps=max_steps,
        render_mode="human",
    )

    print()
    print("=" * 72)
    print(
        f"Random policy rollout  seed={seed}  width={width}  length={length}  "
        f"max_steps={max_steps}"
    )
    print("=" * 72)

    obs, _info = env.reset(seed=seed)
    total_reward = 0.0
    last_reward = 0.0
    print(f"\n--- t=0  reset  obs=({int(obs[0])}, {int(obs[1])}) ---")
    env.render()

    t = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = int(rng.integers(0, env.action_space.n))
        obs, reward, terminated, truncated, _ = env.step(action)
        r = float(reward)
        total_reward += r
        last_reward = r
        t += 1
        name = _ACTION_NAMES[action]
        print(
            f"\n--- t={t}  action={action} ({name})  r={reward}  "
            f"terminated={terminated}  truncated={truncated}  "
            f"obs=({int(obs[0])}, {int(obs[1])}) ---"
        )
        env.render()

    if truncated:
        outcome = "max steps (truncated)"
    elif last_reward == 1.0:
        outcome = "goal (+1)"
    else:
        outcome = "lava (0)"
    print()
    print("-" * 72)
    print(f"Episode finished: {outcome}  steps={t}  return={total_reward}")
    print("-" * 72)


def main() -> int:
    root = _project_root()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # Imports after path fix so `environment` resolves when cwd is not the project root.
    import gymnasium as gym
    import numpy as np

    from environment import TightropeEnv

    print("Python:", sys.version.split()[0])
    print("gymnasium:", gym.__version__)
    print("numpy:", np.__version__)

    env = TightropeEnv(width=5, length=12, max_steps=64, render_mode="ansi")
    assert env.observation_space.shape == (2,)
    assert env.action_space.n == 4

    obs, info = env.reset(seed=0)
    assert isinstance(info, dict)
    assert env.observation_space.contains(obs)
    assert obs.tolist() == [0, 2]  # start x=0, y=width//2

    # Walk along the corridor to the goal (only right moves).
    total_reward = 0.0
    terminated = truncated = False
    for _ in range(20):
        obs, reward, terminated, truncated, _ = env.step(3)
        total_reward += float(reward)
        if terminated or truncated:
            break

    assert terminated and not truncated, "expected goal before max_steps"
    assert obs.tolist()[0] == env.length - 1
    assert total_reward == 1.0

    # Lava: from center row, step "up" until y < 0 terminates with 0 reward.
    env2 = TightropeEnv(width=3, length=10, max_steps=50)
    obs, _ = env2.reset(seed=1)
    assert obs[1] == 1  # width//2 for width=3
    r_sum = 0.0
    done = False
    while not done:
        obs, r, term, trunc, _ = env2.step(0)  # up
        r_sum += r
        done = term or trunc
    assert r_sum == 0.0

    # Invalid action should raise.
    env3 = TightropeEnv(width=3, length=5)
    env3.reset(seed=0)
    try:
        env3.step(99)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for invalid action")

    txt = env.render()
    assert isinstance(txt, str) and "A" in txt and "G" in txt

    # Full episode with random actions; ``human`` mode prints the grid every step.
    _print_random_rollout(seed=7, width=5, length=28, max_steps=120)

    print()
    print("OK: TightropeEnv reset/step/goal/lava/render checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

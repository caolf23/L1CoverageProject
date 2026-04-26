#!/usr/bin/env python3
"""End-to-end smoke run for CODEX.W on ``TightropeEnv``."""

from __future__ import annotations

import sys
from pathlib import Path


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def main() -> int:
    sys.path.insert(0, str(_root()))

    import numpy as np

    from codex.codex_w import run_codex_w
    from environment import TightropeEnv

    def env_factory() -> TightropeEnv:
        return TightropeEnv(width=3, length=8, max_steps=400)

    rng = np.random.default_rng(42)
    covers, policies = run_codex_w(
        env_factory,
        horizon_h=4,
        epsilon=0.34,
        delta=0.05,
        n_actions=4,
        width=3,
        length=8,
        rng=rng,
        epsilon_w=2.0,
        weight_fit_steps=700,
        ppo_rollouts=64,
        ppo_epochs=16,
        ppo_minibatch_size=128,
        ppo_clip_ratio=0.2,
        ppo_lr=1e-3,
        ppo_gamma=1.0,
        ppo_gae_lambda=0.95,
        ppo_value_coef=0.5,
        ppo_entropy_coef=0.0,
        ppo_max_grad_norm=0.5,
        n_weight_cap=96,
        weight_sample_workers=8,
        weight_fit_lr=0.12,
        weight_fit_lr_decay=0.997,
        weight_fit_patience=60,
        weight_zero_absorbing_after_fit=False,
        verbose=True,
    )

    print("CODEX.W smoke run")
    print("  cover layers:", sorted(covers.keys()))
    for h in sorted(policies.keys()):
        print(f"  |policies[{h}]| = {len(policies[h])}")
    for h, mix in covers.items():
        print(f"  p_{h}: {len(mix.policies)} components, weights={np.round(mix.weights, 3)}")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

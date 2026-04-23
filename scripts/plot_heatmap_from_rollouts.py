#!/usr/bin/env python3
"""Plot policy/uniform visit-count heatmaps from saved rollout artifacts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _counts_from_records(
    records: dict[str, np.ndarray], *, width: int, length: int
) -> tuple[np.ndarray, int]:
    counts = np.zeros((width, length), dtype=np.int64)
    absorbing_count = 0
    for x, y in zip(records["x"], records["y"]):
        xi = int(x)
        yi = int(y)
        if xi == length and yi == width:
            absorbing_count += 1
        elif 0 <= xi < length and 0 <= yi < width:
            counts[yi, xi] += 1
    return counts, absorbing_count


def _plot_one(
    *,
    counts: np.ndarray,
    out_path: Path,
    title: str,
    vmax: float,
    colorbar_label: str = "visit count",
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required. Install with `pip install matplotlib`."
        ) from exc

    fig, ax = plt.subplots(figsize=(10, 3.2))
    im = ax.imshow(
        counts,
        origin="lower",
        aspect="auto",
        cmap="magma",
        vmin=0.0,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label=colorbar_label)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_for_h(run_dir: Path, h: int) -> dict[str, object]:
    config_path = run_dir / "config.json"
    rollouts_dir = run_dir / "rollouts"
    figures_dir = run_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    cfg = _load_json(config_path)
    width = int(cfg["width"])
    length = int(cfg["length"])
    horizon_h = int(cfg["horizon_h"])

    mixture_npz = np.load(rollouts_dir / f"h{h}_mixture_rollouts.npz")
    uniform_npz = np.load(rollouts_dir / f"h{h}_uniform_rollouts.npz")
    mixture_records = {k: mixture_npz[k] for k in ("rollout_id", "step", "x", "y")}
    uniform_records = {k: uniform_npz[k] for k in ("rollout_id", "step", "x", "y")}

    mixture_counts, mixture_absorbing = _counts_from_records(
        mixture_records, width=width, length=length
    )
    uniform_counts, uniform_absorbing = _counts_from_records(
        uniform_records, width=width, length=length
    )

    vmax = float(max(np.max(mixture_counts), np.max(uniform_counts), 1))
    policy_path = figures_dir / f"policy_visit_count_h{h}.png"
    uniform_path = figures_dir / f"uniform_visit_count_h{h}.png"
    _plot_one(
        counts=mixture_counts,
        out_path=policy_path,
        title=f"Policy Visit Count (h={h}, absorbing_count={mixture_absorbing})",
        vmax=vmax,
    )
    _plot_one(
        counts=uniform_counts,
        out_path=uniform_path,
        title=f"Uniform Visit Count (h={h}, absorbing_count={uniform_absorbing})",
        vmax=vmax,
    )

    # Log-scale visualization using log(1 + count) so zero-count states stay at 0.
    mixture_log_counts = np.log1p(mixture_counts.astype(np.float64))
    uniform_log_counts = np.log1p(uniform_counts.astype(np.float64))
    vmax_log = float(max(np.max(mixture_log_counts), np.max(uniform_log_counts), 1.0))
    policy_log_path = figures_dir / f"policy_visit_count_log_h{h}.png"
    uniform_log_path = figures_dir / f"uniform_visit_count_log_h{h}.png"
    _plot_one(
        counts=mixture_log_counts,
        out_path=policy_log_path,
        title=f"Policy Visit Count log1p (h={h}, absorbing_count={mixture_absorbing})",
        vmax=vmax_log,
        colorbar_label="log(1 + visit count)",
    )
    _plot_one(
        counts=uniform_log_counts,
        out_path=uniform_log_path,
        title=f"Uniform Visit Count log1p (h={h}, absorbing_count={uniform_absorbing})",
        vmax=vmax_log,
        colorbar_label="log(1 + visit count)",
    )
    return {
        "h": h,
        "policy_path": str(policy_path),
        "uniform_path": str(uniform_path),
        "policy_log_path": str(policy_log_path),
        "uniform_log_path": str(uniform_log_path),
        "mixture_absorbing": int(mixture_absorbing),
        "uniform_absorbing": int(uniform_absorbing),
        "horizon_h": horizon_h,
    }


def plot_all_from_run_dir(run_dir: Path) -> list[dict[str, object]]:
    run_dir = run_dir.expanduser().resolve()
    rollouts_dir = run_dir / "rollouts"
    hs: list[int] = []
    for path in rollouts_dir.glob("h*_mixture_rollouts.npz"):
        m = re.match(r"h(\d+)_mixture_rollouts\.npz", path.name)
        if m:
            hs.append(int(m.group(1)))
    hs = sorted(set(hs))
    outputs: list[dict[str, object]] = []
    for h in hs:
        outputs.append(_plot_for_h(run_dir, h))
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, help="Path to one eval output run dir")
    args = parser.parse_args()
    outputs = plot_all_from_run_dir(Path(args.run_dir))
    if not outputs:
        print("No per-h rollout files found under rollouts/.")
        return 1
    for out in outputs:
        print(f"h={out['h']} policy_heatmap_path: {out['policy_path']}")
        print(f"h={out['h']} uniform_heatmap_path: {out['uniform_path']}")
        print(f"h={out['h']} policy_log_heatmap_path: {out['policy_log_path']}")
        print(f"h={out['h']} uniform_log_heatmap_path: {out['uniform_log_path']}")
        print(
            f"h={out['h']} absorbing_counts: "
            f"mixture={out['mixture_absorbing']} uniform={out['uniform_absorbing']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

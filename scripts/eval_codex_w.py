#!/usr/bin/env python3
"""Evaluate CODEX.W implementation quality beyond smoke-run success."""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

_ROLLOUT_ENV_FACTORY = None
_TABULAR_POLICY_CACHE: dict[int, object] = {}


def _root() -> Path:
    return Path(__file__).resolve().parent.parent


def _structure_checks(
    covers: dict[int, object],
    policies: dict[int, list],
    *,
    horizon_h: int,
    t_rounds: int,
    n_actions: int,
) -> list[str]:
    issues: list[str] = []
    for h in range(1, horizon_h + 1):
        if h not in covers:
            issues.append(f"missing cover p_{h}")
            continue
        mix = covers[h]
        s = float(np.sum(mix.weights))
        if not np.isclose(s, 1.0, atol=1e-6):
            issues.append(f"p_{h} weights sum={s:.6f}, expected 1.0")
        if h >= 2 and len(mix.policies) != t_rounds:
            issues.append(
                f"p_{h} has {len(mix.policies)} components, expected {t_rounds}"
            )
        if h >= 2 and len(policies.get(h, [])) != t_rounds:
            issues.append(
                f"policies[{h}] has {len(policies.get(h, []))}, expected {t_rounds}"
            )

    rng = np.random.default_rng(0)
    probe_obs = np.array([0, 1], dtype=np.int32)
    for h in range(1, horizon_h + 1):
        mix = covers[h]
        for _ in range(8):
            pi = mix.sample_policy(rng)
            a = int(pi.act(probe_obs, 0, rng))
            if not (0 <= a < n_actions):
                issues.append(f"invalid action {a} from p_{h}")
                break
    return issues


def _eval_uniform_policy(
    env_factory,
    *,
    horizon_h: int,
    n_actions: int,
    n_rollouts: int,
    seed: int,
    workers: int = 8,
) -> tuple[int, float, float]:
    """Evaluate pure uniform-random actions for ``horizon_h`` steps."""
    records, success = _collect_rollout_records_uniform(
        env_factory,
        horizon_h=horizon_h,
        n_actions=n_actions,
        n_rollouts=n_rollouts,
        seed=seed,
        workers=workers,
        include_success=True,
    )
    seen = set(zip(records["x"].tolist(), records["y"].tolist()))
    total_steps = int(records["step"].shape[0])
    success_prob = float(success / max(n_rollouts, 1))
    return len(seen), float(len(seen) / max(total_steps, 1)), success_prob


def _eval_mixture_policy(
    env_factory,
    mixture,
    *,
    rollout_steps: int,
    n_rollouts: int,
    seed: int,
    workers: int = 8,
) -> tuple[int, float, float]:
    """Evaluate one cover mixture for a fixed rollout horizon."""
    records, success = _collect_rollout_records_mixture(
        env_factory,
        mixture,
        rollout_steps=rollout_steps,
        n_rollouts=n_rollouts,
        seed=seed,
        workers=workers,
        include_success=True,
    )
    seen = set(zip(records["x"].tolist(), records["y"].tolist()))
    total_steps = int(records["step"].shape[0])
    success_prob = float(success / max(n_rollouts, 1))
    return len(seen), float(len(seen) / max(total_steps, 1)), success_prob


def _cpu_safe_policy(policy, *, verbose: bool = False):
    cache_key = id(policy)
    if cache_key in _TABULAR_POLICY_CACHE:
        return _TABULAR_POLICY_CACHE[cache_key]

    base = getattr(policy, "base", None)
    if base is not None:
        safe_base = _cpu_safe_policy(base, verbose=verbose)
        if safe_base is not base:
            from codex.rollouts import ComposedUniformPolicy

            safe_composed = ComposedUniformPolicy(
                safe_base,
                int(getattr(policy, "n_actions")),
                int(getattr(policy, "first_uniform_timestep")),
            )
            _TABULAR_POLICY_CACHE[cache_key] = safe_composed
            return safe_composed
        _TABULAR_POLICY_CACHE[cache_key] = policy
        return policy
    if hasattr(policy, "to_tabular_policy"):
        t0 = time.perf_counter()
        out = policy.to_tabular_policy()
        if verbose:
            print(
                f"[timing] extract tabular policy ({type(policy).__name__}): "
                f"{time.perf_counter() - t0:.3f}s"
            )
        _TABULAR_POLICY_CACHE[cache_key] = out
        return out
    _TABULAR_POLICY_CACHE[cache_key] = policy
    return policy


def _pool_for_workers(workers: int):
    if workers <= 1:
        return None
    try:
        return mp.get_context("fork").Pool(processes=workers)
    except ValueError:
        return mp.get_context().Pool(processes=workers)


def _sample_component_index(weights: np.ndarray, rng: np.random.Generator) -> int:
    return int(rng.choice(len(weights), p=weights))


def _uniform_rollout_worker(args):
    rollout_id, horizon_h, n_actions, seed = args
    env = _ROLLOUT_ENV_FACTORY()
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    terminated = truncated = False
    reached_goal = False
    rec_step: list[int] = []
    rec_x: list[int] = []
    rec_y: list[int] = []
    for step in range(horizon_h):
        x, y = int(obs[0]), int(obs[1])
        rec_step.append(step)
        rec_x.append(x)
        rec_y.append(y)
        if terminated or truncated:
            break
        a = int(rng.integers(0, n_actions))
        obs, reward, terminated, truncated, _ = env.step(a)
        if float(reward) == 1.0:
            reached_goal = True
    return rollout_id, rec_step, rec_x, rec_y, int(reached_goal)


def _mixture_rollout_worker(args):
    rollout_id, policy, rollout_steps, seed = args
    env = _ROLLOUT_ENV_FACTORY()
    rng = np.random.default_rng(seed)
    obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    terminated = truncated = False
    reached_goal = False
    rec_step: list[int] = []
    rec_x: list[int] = []
    rec_y: list[int] = []
    for step in range(rollout_steps):
        x, y = int(obs[0]), int(obs[1])
        rec_step.append(step)
        rec_x.append(x)
        rec_y.append(y)
        if terminated or truncated:
            break
        a = int(policy.act(np.asarray(obs), step, rng))
        obs, reward, terminated, truncated, _ = env.step(a)
        if float(reward) == 1.0:
            reached_goal = True
    return rollout_id, rec_step, rec_x, rec_y, int(reached_goal)


def _collect_rollout_records_uniform(
    env_factory,
    *,
    horizon_h: int,
    n_actions: int,
    n_rollouts: int,
    seed: int,
    workers: int = 8,
    include_success: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], int]:
    """Collect raw per-step rollout records for pure uniform-random policy."""
    rng = np.random.default_rng(seed)
    global _ROLLOUT_ENV_FACTORY
    _ROLLOUT_ENV_FACTORY = env_factory
    seeds = [int(x) for x in rng.integers(0, 2**31 - 1, size=n_rollouts, dtype=np.int64)]
    tasks = [(rid, horizon_h, n_actions, seeds[rid]) for rid in range(n_rollouts)]
    pool = _pool_for_workers(max(int(workers), 1))
    try:
        if pool is None:
            results = [_uniform_rollout_worker(t) for t in tasks]
        else:
            results = pool.map(_uniform_rollout_worker, tasks)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _ROLLOUT_ENV_FACTORY = None
    records_rollout_id: list[int] = []
    records_step: list[int] = []
    records_x: list[int] = []
    records_y: list[int] = []
    success = 0
    for rid, step_list, x_list, y_list, ok in results:
        success += int(ok)
        records_rollout_id.extend([rid] * len(step_list))
        records_step.extend(step_list)
        records_x.extend(x_list)
        records_y.extend(y_list)
    payload = {
        "rollout_id": np.asarray(records_rollout_id, dtype=np.int32),
        "step": np.asarray(records_step, dtype=np.int32),
        "x": np.asarray(records_x, dtype=np.int32),
        "y": np.asarray(records_y, dtype=np.int32),
    }
    if include_success:
        return payload, int(success)
    return payload


def _collect_rollout_records_mixture(
    env_factory,
    mixture,
    *,
    rollout_steps: int,
    n_rollouts: int,
    seed: int,
    workers: int = 8,
    include_success: bool = False,
    verbose: bool = False,
) -> dict[str, np.ndarray] | tuple[dict[str, np.ndarray], int]:
    """Collect raw per-step rollout records for one cover mixture."""
    rng = np.random.default_rng(seed)
    global _ROLLOUT_ENV_FACTORY
    _ROLLOUT_ENV_FACTORY = env_factory
    cache_t0 = time.perf_counter()
    cached_components = [
        _cpu_safe_policy(policy, verbose=verbose) for policy in mixture.policies
    ]
    cache_elapsed = time.perf_counter() - cache_t0
    sampled_component_indices = [
        _sample_component_index(mixture.weights, rng) for _ in range(n_rollouts)
    ]
    rollout_policies = [cached_components[idx] for idx in sampled_component_indices]
    if verbose:
        print(
            "[timing] mixture component cache build: "
            f"{cache_elapsed:.3f}s "
            f"(components={len(cached_components)}, rollouts={n_rollouts}, "
            f"reuse_hits={max(n_rollouts - len(cached_components), 0)})"
        )
    seeds = [int(x) for x in rng.integers(0, 2**31 - 1, size=n_rollouts, dtype=np.int64)]
    tasks = [
        (rid, rollout_policies[rid], rollout_steps, seeds[rid])
        for rid in range(n_rollouts)
    ]
    pool = _pool_for_workers(max(int(workers), 1))
    try:
        if pool is None:
            results = [_mixture_rollout_worker(t) for t in tasks]
        else:
            results = pool.map(_mixture_rollout_worker, tasks)
    finally:
        if pool is not None:
            pool.close()
            pool.join()
        _ROLLOUT_ENV_FACTORY = None
    records_rollout_id: list[int] = []
    records_step: list[int] = []
    records_x: list[int] = []
    records_y: list[int] = []
    success = 0
    for rid, step_list, x_list, y_list, ok in results:
        success += int(ok)
        records_rollout_id.extend([rid] * len(step_list))
        records_step.extend(step_list)
        records_x.extend(x_list)
        records_y.extend(y_list)
    payload = {
        "rollout_id": np.asarray(records_rollout_id, dtype=np.int32),
        "step": np.asarray(records_step, dtype=np.int32),
        "x": np.asarray(records_x, dtype=np.int32),
        "y": np.asarray(records_y, dtype=np.int32),
    }
    if include_success:
        return payload, int(success)
    return payload


def _make_run_dirs(root_dir: Path, width: int, length: int, t_rounds: int) -> dict[str, Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{ts}_w{width}_l{length}_t{t_rounds}"
    run_dir = root_dir / "output" / run_id
    dirs = {
        "run_dir": run_dir,
        "artifacts_dir": run_dir / "artifacts",
        "rollouts_dir": run_dir / "rollouts",
        "figures_dir": run_dir / "figures",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def _save_json(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(_to_jsonable(payload), f, ensure_ascii=False, indent=2)


def _flatten_numeric_values(obj) -> list[int]:
    vals: list[int] = []
    if isinstance(obj, (list, tuple)):
        for v in obj:
            vals.extend(_flatten_numeric_values(v))
        return vals
    if isinstance(obj, np.ndarray):
        return _flatten_numeric_values(obj.tolist())
    if isinstance(obj, (np.integer, int)):
        return [int(obj)]
    if isinstance(obj, (np.floating, float)):
        return [int(obj)]
    return vals


def _export_final_mixture(
    mixture,
    *,
    json_path: Path,
    npz_path: Path,
    weights_dir: Path | None = None,
    verbose: bool = False,
) -> None:
    component_payloads: list[dict] = []
    rows: list[tuple[int, int, int, int, float]] = []
    for comp_idx, policy in enumerate(mixture.policies):
        payload: dict[str, object] = {
            "component_index": comp_idx,
            "type": type(policy).__name__,
        }
        if hasattr(policy, "first_uniform_timestep"):
            payload["first_uniform_timestep"] = int(
                getattr(policy, "first_uniform_timestep")
            )
        base = getattr(policy, "base", None)
        if base is not None:
            safe_base = _cpu_safe_policy(base, verbose=verbose)
            if safe_base is not base:
                payload["base_tabularized"] = True
            payload["base_type"] = type(base).__name__
            if hasattr(base, "metadata"):
                payload["base_metadata"] = _to_jsonable(base.metadata())
            if isinstance(weights_dir, Path) and hasattr(base, "save"):
                weights_dir.mkdir(parents=True, exist_ok=True)
                weights_path = weights_dir / f"component_{comp_idx}_base_policy.pt"
                base.save(weights_path)
                payload["base_weights_path"] = str(weights_path)
            training_stats = getattr(base, "training_stats", None)
            if isinstance(training_stats, dict):
                payload["base_training_stats"] = _to_jsonable(training_stats)
            probs = getattr(safe_base, "probs", getattr(base, "probs", None))
        else:
            safe_policy = _cpu_safe_policy(policy, verbose=verbose)
            if safe_policy is not policy:
                payload["tabularized"] = True
            if hasattr(policy, "metadata"):
                payload["metadata"] = _to_jsonable(policy.metadata())
            if isinstance(weights_dir, Path) and hasattr(policy, "save"):
                weights_dir.mkdir(parents=True, exist_ok=True)
                weights_path = weights_dir / f"component_{comp_idx}_policy.pt"
                policy.save(weights_path)
                payload["weights_path"] = str(weights_path)
            training_stats = getattr(policy, "training_stats", None)
            if isinstance(training_stats, dict):
                payload["training_stats"] = _to_jsonable(training_stats)
            probs = getattr(safe_policy, "probs", getattr(policy, "probs", None))
        if isinstance(probs, dict):
            exported_probs = {}
            for sk, prob_vec in probs.items():
                flat_vals = _flatten_numeric_values(sk)
                if flat_vals:
                    key = ",".join(str(v) for v in flat_vals)
                else:
                    key = str(sk)
                exported_probs[key] = [float(x) for x in np.asarray(prob_vec).tolist()]
                if len(flat_vals) >= 2:
                    sx, sy = int(flat_vals[0]), int(flat_vals[1])
                    for a_idx, p in enumerate(np.asarray(prob_vec, dtype=np.float64)):
                        rows.append((comp_idx, sx, sy, int(a_idx), float(p)))
            payload["tabular_probs"] = exported_probs
        component_payloads.append(payload)
    _save_json(
        json_path,
        {
            "weights": np.asarray(mixture.weights, dtype=np.float64).tolist(),
            "components": component_payloads,
        },
    )
    if rows:
        arr = np.asarray(rows, dtype=np.float64)
        np.savez_compressed(
            npz_path,
            weights=np.asarray(mixture.weights, dtype=np.float64),
            component_index=arr[:, 0].astype(np.int32),
            state_x=arr[:, 1].astype(np.int32),
            state_y=arr[:, 2].astype(np.int32),
            action=arr[:, 3].astype(np.int32),
            prob=arr[:, 4].astype(np.float64),
        )
    else:
        np.savez_compressed(npz_path, weights=np.asarray(mixture.weights, dtype=np.float64))


def _save_rollout_records(
    records: dict[str, np.ndarray],
    *,
    source: str,
    json_path: Path,
    npz_path: Path,
    h: int,
    n_rollouts: int,
    rollout_steps: int,
) -> None:
    np.savez_compressed(npz_path, **records)
    payload = {
        "source": source,
        "h": int(h),
        "n_rollouts": int(n_rollouts),
        "rollout_steps": int(rollout_steps),
        "num_records": int(records["x"].shape[0]),
        "rollout_id": records["rollout_id"].tolist(),
        "step": records["step"].tolist(),
        "x": records["x"].tolist(),
        "y": records["y"].tolist(),
    }
    _save_json(json_path, payload)


def _semantic_checks(
    env_factory,
    *,
    width: int,
    length: int,
    n_actions: int,
) -> list[str]:
    """Validate absorbing-state semantics and weight parameterization."""
    issues: list[str] = []

    from codex.rollouts import tightrope_predict_next
    from models.weight_fn import TabularWeightModel

    env = env_factory()
    obs, _ = env.reset(seed=123)
    # width=2 starts at y=1, action=1 moves to y=2 -> lava -> absorbing.
    obs_absorb, reward, terminated, truncated, _ = env.step(1)
    expected_absorb = np.array([length, width], dtype=np.int32)
    if not np.array_equal(obs_absorb, expected_absorb):
        issues.append(
            f"env did not enter absorbing state on lava: got={obs_absorb}, expected={expected_absorb}"
        )
    if terminated:
        issues.append("absorbing transition unexpectedly sets terminated=True")
    if truncated:
        issues.append("absorbing transition unexpectedly sets truncated=True")
    if float(reward) != 0.0:
        issues.append(f"absorbing transition reward expected 0.0, got {float(reward)}")

    for a in range(n_actions):
        obs_loop, rew_loop, term_loop, trunc_loop, _ = env.step(a)
        if not np.array_equal(obs_loop, expected_absorb):
            issues.append(f"absorbing self-loop broken for action={a}, got={obs_loop}")
            break
        if term_loop:
            issues.append(f"absorbing self-loop sets terminated=True for action={a}")
            break
        if float(rew_loop) != 0.0:
            issues.append(f"absorbing self-loop reward not zero for action={a}")
            break
        if trunc_loop:
            break

    p_next, term_pred, trunc_pred, r_pred = tightrope_predict_next(
        np.asarray(obs, dtype=np.int32), 1, width, length
    )
    if not np.array_equal(p_next, expected_absorb):
        issues.append(
            f"predictor mismatch for lava transition: got={p_next}, expected={expected_absorb}"
        )
    if term_pred or trunc_pred or float(r_pred) != 0.0:
        issues.append(
            "predictor lava transition expected (terminated=False,truncated=False,reward=0.0)"
        )

    model = TabularWeightModel(length=length, width=width, n_actions=n_actions)
    n_states = model.n_states
    sum_w = 0.0
    test_obs = np.array([0, min(width - 1, 1)], dtype=np.int32)
    for x in range(length):
        for y in range(width):
            sum_w += model.prob(test_obs, 0, np.array([x, y], dtype=np.int32))
    sum_w += model.prob(test_obs, 0, np.array([length, width], dtype=np.int32))
    if np.isclose(sum_w, 1.0, atol=1e-2):
        issues.append(
            f"weight model still appears softmax-normalized (sum over next states ~= 1, got {sum_w:.6f})"
        )
    if not (0.0 < sum_w < float(n_states)):
        issues.append(f"weight sum over next states out of expected sigmoid range: {sum_w:.6f}")

    return issues


def main() -> int:
    sys.path.insert(0, str(_root()))

    from codex.codex_w import run_codex_w
    from environment import TightropeEnv

    # width, length = 5, 100
    # n_actions = 4
    # horizon_h = 200
    # epsilon = 0.1

    width, length = 1, 15
    n_actions = 4
    horizon_h = 10
    epsilon = 0.1

    t_rounds = int(np.ceil(1.0 / epsilon))
    root_dir = _root()
    run_dirs = _make_run_dirs(root_dir, width=width, length=length, t_rounds=t_rounds)

    def env_factory() -> TightropeEnv:
        return TightropeEnv(width=width, length=length, max_steps=400)

    ppo_clip_ratio = 0.2
    per_h_rollouts = 500
    rollout_workers = 8
    print("ppo_clip_ratio: ", ppo_clip_ratio)
    print("rollout_workers: ", rollout_workers)
    rng_seed = 42
    run_codex_w_kwargs = {
        "horizon_h": horizon_h,
        "epsilon": epsilon,
        "delta": 0.05,
        "n_actions": n_actions,
        "width": width,
        "length": length,
        "rng": np.random.default_rng(rng_seed),
        "epsilon_w": 0.5,
        "weight_fit_steps": 500,
        "ppo_rollouts": 2048,
        "ppo_epochs": 32,
        "ppo_minibatch_size": 128,
        "ppo_clip_ratio": ppo_clip_ratio,
        "ppo_lr": 2e-3,
        "ppo_gamma": 1.0,
        "ppo_gae_lambda": 1.0,
        "ppo_value_coef": 0.5,
        "ppo_entropy_coef": 0.0,
        "ppo_max_grad_norm": 1.0,
        "n_weight_cap": 1024,
        "weight_sample_workers": 8,
        "weight_fit_lr": 0.50,
        "weight_fit_lr_decay": 1.0,
        "weight_fit_patience": 300,
        "weight_zero_absorbing_after_fit": False,
        "return_diagnostics": True,
        "verbose": True,
    }
    verbose = bool(run_codex_w_kwargs.get("verbose", False))
    config_payload = {
        "script": "scripts/eval_codex_w.py",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "width": width,
        "length": length,
        "horizon_h": horizon_h,
        "epsilon": epsilon,
        "t_rounds": t_rounds,
        "seed": rng_seed,
        "run_codex_w_kwargs": {
            k: ("np.random.default_rng(seed=42)" if k == "rng" else v)
            for k, v in run_codex_w_kwargs.items()
        },
        "per_h_rollouts": per_h_rollouts,
        "rollout_workers": rollout_workers,
    }
    config_path = run_dirs["run_dir"] / "config.json"
    _save_json(config_path, config_payload)
    print(f"run_output_dir: {run_dirs['run_dir'].resolve()}")
    print(f"config_path: {config_path.resolve()}")

    def _on_layer_complete(h: int, cover) -> None:
        cb_t0 = time.perf_counter()
        mixture_records_h = _collect_rollout_records_mixture(
            env_factory,
            cover,
            rollout_steps=h,
            n_rollouts=per_h_rollouts,
            seed=2700 + h,
            workers=rollout_workers,
            verbose=verbose,
        )
        uniform_records_h = _collect_rollout_records_uniform(
            env_factory,
            horizon_h=h,
            n_actions=n_actions,
            n_rollouts=per_h_rollouts,
            seed=2900 + h,
            workers=rollout_workers,
        )
        mixture_rollouts_json_h = run_dirs["rollouts_dir"] / f"h{h}_mixture_rollouts.json"
        mixture_rollouts_npz_h = run_dirs["rollouts_dir"] / f"h{h}_mixture_rollouts.npz"
        uniform_rollouts_json_h = run_dirs["rollouts_dir"] / f"h{h}_uniform_rollouts.json"
        uniform_rollouts_npz_h = run_dirs["rollouts_dir"] / f"h{h}_uniform_rollouts.npz"
        _save_rollout_records(
            mixture_records_h,
            source="mixture",
            json_path=mixture_rollouts_json_h,
            npz_path=mixture_rollouts_npz_h,
            h=h,
            n_rollouts=per_h_rollouts,
            rollout_steps=h,
        )
        _save_rollout_records(
            uniform_records_h,
            source="uniform",
            json_path=uniform_rollouts_json_h,
            npz_path=uniform_rollouts_npz_h,
            h=h,
            n_rollouts=per_h_rollouts,
            rollout_steps=h,
        )
        print(
            f"[realtime] h={h} saved "
            f"mixture_records={int(mixture_records_h['x'].shape[0])} "
            f"uniform_records={int(uniform_records_h['x'].shape[0])}"
        )
        print(f"[realtime] h={h} mixture_npz={mixture_rollouts_npz_h.resolve()}")
        print(f"[realtime] h={h} uniform_npz={uniform_rollouts_npz_h.resolve()}")
        if verbose:
            print(
                f"[timing] rollout policy & save (h={h}): "
                f"{time.perf_counter() - cb_t0:.3f}s"
            )

    covers, policies, diagnostics = run_codex_w(
        env_factory,
        on_layer_complete=_on_layer_complete,
        **run_codex_w_kwargs,
    )

    print("=== Structure Checks ===")
    issues = _structure_checks(
        covers, policies, horizon_h=horizon_h, t_rounds=t_rounds, n_actions=n_actions
    )
    issues.extend(
        _semantic_checks(
            env_factory,
            width=width,
            length=length,
            n_actions=n_actions,
        )
    )
    if issues:
        for msg in issues:
            print("FAIL:", msg)
    else:
        print("PASS: all structure checks passed.")

    print("\n=== Algorithm-5 Objective Trend (per (h,t)) ===")
    improved = 0
    for d in diagnostics:
        h = int(d["layer"])
        t = int(d["iteration"])
        jb = d["objective_before"]
        ja = d["objective_after"]
        delta = ja - jb
        if delta >= 0:
            improved += 1
        print(
            f"h={h} t={t} n_d1={int(d['n_d1'])} n_d2={int(d['n_d2'])} "
            f"J_before={jb:.6f} J_after={ja:.6f} dJ={delta:+.6f}"
        )
    print(f"Improved entries: {improved}/{len(diagnostics)}")

    print("\n=== Per-Layer p_i vs i-step Uniform ===")
    for i in range(1, horizon_h + 1):
        p_uniq, p_density, p_success = _eval_mixture_policy(
            env_factory,
            covers[i],
            rollout_steps=i,
            n_rollouts=per_h_rollouts,
            seed=700 + i,
            workers=rollout_workers,
        )
        u_uniq, u_density, u_success = _eval_uniform_policy(
            env_factory,
            horizon_h=i,
            n_actions=n_actions,
            n_rollouts=per_h_rollouts,
            seed=900 + i,
            workers=rollout_workers,
        )
        print(
            f"i={i}: "
            f"p_i(unique={p_uniq}, uniq_per_step={p_density:.4f}, success={p_success:.4f})  "
            f"uniform_i(unique={u_uniq}, uniq_per_step={u_density:.4f}, success={u_success:.4f})"
        )

    print("\n=== Export Final Mixture + Rollout Data ===")
    final_mixture = covers[horizon_h]
    final_mixture_json = run_dirs["artifacts_dir"] / "final_mixture_policy.json"
    final_mixture_npz = run_dirs["artifacts_dir"] / "final_mixture_policy.npz"
    final_mixture_weights_dir = run_dirs["artifacts_dir"] / "final_mixture_weights"
    export_t0 = time.perf_counter()
    _export_final_mixture(
        final_mixture,
        json_path=final_mixture_json,
        npz_path=final_mixture_npz,
        weights_dir=final_mixture_weights_dir,
        verbose=verbose,
    )
    if verbose:
        print(
            f"[timing] extract tabular policy + export final mixture: "
            f"{time.perf_counter() - export_t0:.3f}s"
        )
    print(f"final_mixture_json: {final_mixture_json.resolve()}")
    print(f"final_mixture_npz: {final_mixture_npz.resolve()}")

    print(
        "next_step_plot_cmd: "
        f"python scripts/plot_heatmap_from_rollouts.py --run-dir {run_dirs['run_dir']}"
    )
    try:
        from scripts.plot_heatmap_from_rollouts import plot_all_from_run_dir

        plot_outputs = plot_all_from_run_dir(run_dirs["run_dir"], count_mode="both")
        print(f"auto_plot_generated_files: {len(plot_outputs)}")
    except Exception as exc:
        print(f"WARN: auto plot generation failed: {exc}")

    if issues:
        print("\nOVERALL: check FAILED (see FAIL lines).")
        return 1

    print("\nOVERALL: check PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

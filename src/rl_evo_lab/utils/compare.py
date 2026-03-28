"""Run experiment suites across multiple seeds and plot comparison.

Each condition is defined as a (label, EDERConfig) pair. All runs are
dispatched in parallel via ProcessPoolExecutor so seeds × conditions run
concurrently.

Usage:
    poetry run python -m rl_evo_lab.utils.compare                              # default suite, cartpole
    poetry run python -m rl_evo_lab.utils.compare --seeds 1 2 3
    poetry run python -m rl_evo_lab.utils.compare --experiment efficiency --env lunarlander
    poetry run python -m rl_evo_lab.utils.compare --workers 4
    poetry run python -m rl_evo_lab.utils.compare --show
    poetry run python -m rl_evo_lab.utils.compare --plot-only --show           # skip training

Available experiments: eder_vs_baseline, efficiency, model_size, updates, sample_efficiency
"""
from __future__ import annotations

import argparse
import hashlib
import multiprocessing
import os
import queue
import shutil
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

_console = Console()

from rl_evo_lab.train import train
from rl_evo_lab.utils.config import EDERConfig, ENV_PRESETS, make_config
from rl_evo_lab.utils.logging import _run_id


_RUNS_DIR = "runs"

# ---------------------------------------------------------------------------
# Condition + Experiment types
# ---------------------------------------------------------------------------

# (label, EDERConfig overrides relative to env preset + experiment base_overrides)
ConditionSpec = tuple[str, dict[str, Any]]


@dataclass
class ExperimentSpec:
    """Self-contained description of one experiment.

    conditions:   list of (label, config_overrides) — the algorithm variants to compare.
    env_overrides: per-env HP overrides applied on top of ENV_PRESET and before condition
                   overrides. Use to make an experiment tractable on a specific env without
                   touching the global preset (e.g. fewer episodes for a quick smoke test).
                   These values are hashed into run IDs so changing them invalidates old runs.

    Example::

        ExperimentSpec(
            conditions=[("EDER", {"use_novelty": True}), ("DQN", {"use_es": False})],
            env_overrides={"lunarlander": {"total_episodes": 500}},
        )
    """
    conditions: list[ConditionSpec]
    env_overrides: dict[str, dict[str, Any]] = field(default_factory=dict)


EXPERIMENTS: dict[str, ExperimentSpec] = {
    # Original: does novelty help ES?
    "eder_vs_baseline": ExperimentSpec(
        conditions=[
            ("EDER",     {"use_novelty": True}),
            ("Baseline", {"use_novelty": False}),
        ],
    ),
    # Does ES add value vs pure DQN?
    "efficiency": ExperimentSpec(
        conditions=[
            ("EDER",   {"use_es": True,  "use_novelty": True}),
            ("ES+DQN", {"use_es": True,  "use_novelty": False}),
            ("DQN",    {"use_es": False, "use_novelty": False}),
        ],
    ),
    # Smaller model — does ES compensate?
    "model_size": ExperimentSpec(
        conditions=[
            ("EDER-64",  {"use_es": True,  "use_novelty": True,  "hidden_dim": 64}),
            ("EDER-128", {"use_es": True,  "use_novelty": True,  "hidden_dim": 128}),
            ("DQN-64",   {"use_es": False, "use_novelty": False, "hidden_dim": 64}),
            ("DQN-128",  {"use_es": False, "use_novelty": False, "hidden_dim": 128}),
        ],
    ),
    # Fewer learner updates — can ES + novelty compensate?
    "updates": ExperimentSpec(
        conditions=[
            ("EDER-5upd",  {"use_es": True,  "use_novelty": True,  "learner_updates_per_episode": 5}),
            ("EDER-20upd", {"use_es": True,  "use_novelty": True,  "learner_updates_per_episode": 20}),
            ("DQN-5upd",   {"use_es": False, "use_novelty": False, "learner_updates_per_episode": 5}),
            ("DQN-20upd",  {"use_es": False, "use_novelty": False, "learner_updates_per_episode": 20}),
        ],
    ),
    # Fair comparison: ES uses N× more env steps per episode than DQN.
    # DQN gets proportionally more episodes so all conditions have a comparable env-step budget.
    # Defaults to env_steps x-axis to make true sample cost visible.
    "sample_efficiency": ExperimentSpec(
        conditions=[
            ("EDER",   {"use_es": True,  "use_novelty": True}),
            ("ES+DQN", {"use_es": True,  "use_novelty": False}),
            # DQN gets 20× more episodes to match the env-step budget of 20 ES workers.
            ("DQN",    {"use_es": False, "use_novelty": False, "total_episodes": 10_000}),
        ],
    ),
}


def _make_config_for(env: str, spec: ExperimentSpec, seed: int, condition_overrides: dict[str, Any]) -> EDERConfig:
    """Build a config applying: env preset → experiment env_overrides → condition overrides."""
    env_extra = spec.env_overrides.get(env, {})
    return make_config(env, seed=seed, **{**env_extra, **condition_overrides})


def _build_conditions(experiment: str, env: str, seeds: list[int]) -> dict[str, list[Path]]:
    """Return {label: [csv_path_per_seed]} without running training."""
    spec = EXPERIMENTS[experiment]
    paths: dict[str, list[Path]] = {}
    for label, overrides in spec.conditions:
        paths[label] = [
            Path(_RUNS_DIR) / _run_id(_make_config_for(env, spec, s, overrides)) / "metrics.csv"
            for s in seeds
        ]
    return paths


def _train_worker(args: tuple) -> tuple[str, Path]:
    """Top-level function required for ProcessPoolExecutor pickling."""
    cfg, q, force = args
    run_dir = Path(_RUNS_DIR) / _run_id(cfg)
    csv = run_dir / "metrics.csv"
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    if not csv.exists():
        train(cfg, log_dir=_RUNS_DIR, verbose=False, progress_queue=q)
    return _run_id(cfg), csv


def _compare_digest(paths: dict[str, list[Path]]) -> str:
    """SHA-1 of all run IDs in the comparison (sorted for stability)."""
    all_ids = sorted(str(p.parent.name) for csv_list in paths.values() for p in csv_list)
    blob = "\n".join(all_ids).encode()
    return hashlib.sha1(blob).hexdigest()[:8]


def _compare_dir(seeds: list[int], experiment: str, env: str, paths: dict[str, list[Path]]) -> Path:
    """Content-addressed directory: same configs → same dir, any config change → new dir."""
    seed_key = "_".join(str(s) for s in sorted(seeds))
    digest = _compare_digest(paths)
    d = Path(_RUNS_DIR) / f"compare__{experiment}__{env}__seeds_{seed_key}__{digest}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _condition_label(cfg: EDERConfig) -> str:
    if cfg.use_es:
        return "EDER" if cfg.use_novelty else "ES+DQN"
    return "DQN"


def _run_condition(
    label: str,
    cfgs: list[EDERConfig],
    max_workers: int | None,
    force: bool = False,
) -> dict[str, Path]:
    """Run one condition's seeds in parallel. Returns {run_id: csv_path}."""
    def _is_cached(cfg: EDERConfig) -> bool:
        return not force and (Path(_RUNS_DIR) / _run_id(cfg) / "metrics.csv").exists()

    pending = [cfg for cfg in cfgs if not _is_cached(cfg)]
    result: dict[str, Path] = {
        _run_id(cfg): Path(_RUNS_DIR) / _run_id(cfg) / "metrics.csv"
        for cfg in cfgs if _is_cached(cfg)
    }

    if not pending:
        _console.print(f"  [dim]{label}[/dim] — all seeds cached")
        return result

    n_workers = min(len(pending), max_workers or len(pending))

    progress = Progress(
        TextColumn("  [cyan]{task.description:<22}[/cyan]"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[stats]}"),
    )
    task_ids = {
        _run_id(cfg): progress.add_task(f"seed={cfg.seed}", total=cfg.total_episodes, stats="")
        for cfg in pending
    }

    manager = multiprocessing.Manager()
    q = manager.Queue()

    def _listen() -> None:
        while True:
            try:
                msg = q.get(timeout=0.5)
                if msg is None:
                    break
                rid = msg["run_id"]
                if rid not in task_ids:
                    continue
                parts = [f"loss={msg['loss']:.4f}", f"buf={msg['buf']:,}"]
                if msg.get("beta"):
                    parts.append(f"β={msg['beta']:.4f}")
                if msg.get("eval") is not None:
                    parts.append(f"[green]eval={msg['eval']:.1f}[/green]")
                if msg.get("sync"):
                    parts.append("[yellow]sync[/yellow]")
                progress.update(task_ids[rid], completed=msg["episode"] + 1, stats="  ".join(parts))
            except queue.Empty:
                continue

    listener = threading.Thread(target=_listen, daemon=True)

    with Live(progress, refresh_per_second=10, console=_console):
        listener.start()
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_train_worker, (cfg, q, force)): cfg for cfg in pending}
            for future in as_completed(futures):
                run_id, csv = future.result()
                result[run_id] = csv
                if run_id in task_ids:
                    progress.update(task_ids[run_id], completed=futures[future].total_episodes)
        q.put(None)
        listener.join()
    manager.shutdown()

    return result


def run_all(
    experiment: str,
    env: str,
    seeds: list[int],
    max_workers: int | None = None,
    force: bool = False,
) -> dict[str, list[Path]]:
    """Run conditions sequentially, seeds within each condition in parallel.

    Prints per-condition wall-clock time so you can compare EDER vs ES+DQN vs DQN speed.
    force=True deletes and re-runs any existing runs for this experiment.
    """
    spec = EXPERIMENTS[experiment]
    cfg_to_path: dict[str, Path] = {}

    total_conditions = len(spec.conditions)
    for i, (label, overrides) in enumerate(spec.conditions, 1):
        cfgs = [_make_config_for(env, spec, s, overrides) for s in seeds]
        _console.rule(f"[bold]{label}[/bold]  ({i}/{total_conditions})")
        t0 = time.monotonic()
        cfg_to_path.update(_run_condition(label, cfgs, max_workers, force=force))
        elapsed = time.monotonic() - t0
        mins, secs = divmod(int(elapsed), 60)
        _console.print(f"  [bold green]✓[/bold green] {label} — {mins}m {secs:02d}s\n")

    paths: dict[str, list[Path]] = {}
    for label, overrides in spec.conditions:
        paths[label] = [
            cfg_to_path[_run_id(_make_config_for(env, spec, s, overrides))]
            for s in seeds
        ]
    return paths


def _smooth(arr: np.ndarray, window: int = 15) -> np.ndarray:
    """Centered rolling mean over a 1-D array."""
    out = np.empty_like(arr)
    half = window // 2
    for i in range(len(arr)):
        lo, hi = max(0, i - half), min(len(arr), i + half + 1)
        out[i] = arr[lo:hi].mean()
    return out


def _aggregate(
    csv_list: list[Path],
    col: str,
    smooth: bool = False,
    x_col: str = "episode",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (x_values, mean, std) aggregated across seeds.

    x_col: column to use as x-axis — "episode" (default) or "total_env_steps".
    For sparse columns (those with NaN rows) values are forward-filled.
    When x_col="total_env_steps", x values are taken from the first seed's
    column (all seeds share the same episode count, steps may differ slightly).
    """
    dfs = [pd.read_csv(p) for p in csv_list]
    n = min(len(d) for d in dfs)
    dfs = [d.iloc[:n] for d in dfs]

    x = dfs[0][x_col].values[:n] if x_col != "episode" else dfs[0]["episode"].values + 1

    if dfs[0][col].isna().any():
        arrays = [d[col].ffill().bfill().values[:n] for d in dfs]
    else:
        arrays = [d[col].values[:n] for d in dfs]

    stacked = np.stack(arrays)
    mean = stacked.mean(axis=0)
    std  = stacked.std(axis=0)

    if smooth:
        mean = _smooth(mean)
        std  = _smooth(std)

    return x, mean, std


_PALETTE = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink"]
_LINESTYLES = ["-", "--", "-.", ":"]

# "Solved" thresholds per env_id — drawn as a horizontal reference line.
# Values are the reward at which the environment is considered solved.
_SOLVED_THRESHOLDS: dict[str, float] = {
    "CartPole-v1":    475.0,
    "LunarLander-v3": 200.0,
    "Acrobot-v1":    -100.0,
    "MountainCar-v0": -110.0,
}

# Panel config: (csv_column, y_label, subtitle, smooth, higher_is_better)
_PANELS_BASE = [
    ("actor_extrinsic_reward", "reward", "ES Worker Return\n(mean across population, higher = better exploration)",    True,  True),
    ("learner_eval_reward",    "reward", "Learner Eval Reward\n(greedy policy, no noise — the main performance metric)", False, True),
    ("learner_loss",           "loss",   "DQN Loss\n(Huber, log scale — lower = more stable Q-values)",                True,  False),
]
_PANEL_BETA = (
    "effective_beta", "β",
    "Effective Novelty Weight β\n(zero during warmup, rises as IDN learns)",
    False, None,
)
_PANEL_DIVERSITY = (
    "buffer_diversity", "diversity",
    "Replay Buffer Diversity\n(mean pairwise distance of sampled obs — higher = more varied experience)",
    False, True,
)


def _detect_env(csv_list: list[Path]) -> str:
    """Infer env_id from the run directory name."""
    name = csv_list[0].parent.name  # e.g. "CartPole-v1__seed42__..."
    return name.split("__")[0]


def _any_novelty(paths: dict[str, list[Path]]) -> bool:
    """Return True if at least one condition has non-zero effective_beta values."""
    for csv_list in paths.values():
        try:
            df = pd.read_csv(csv_list[0])
            if "effective_beta" in df.columns and df["effective_beta"].max() > 0:
                return True
        except Exception:
            pass
    return False


def compare(
    paths: dict[str, list[Path]],
    out_dir: Path | None = None,
    show: bool = False,
    title: str = "",
    x_col: str = "episode",
) -> Path:
    """Plot mean ± std band per condition across all seeds.

    x_col: "episode" (default) or "total_env_steps" for fair ES vs DQN comparison.
    """
    first_csvs = next(iter(paths.values()))
    env_id = _detect_env(first_csvs)
    solved = _SOLVED_THRESHOLDS.get(env_id)

    # 4th panel: β schedule when novelty is used, buffer diversity otherwise
    fourth_panel = _PANEL_BETA if _any_novelty(paths) else _PANEL_DIVERSITY
    panels = _PANELS_BASE + [fourth_panel]

    x_label = "Env Steps" if x_col == "total_env_steps" else "Episode"

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(title or "Condition comparison — mean ± std across seeds", fontsize=12, fontweight="bold")
    ax_map = dict(zip([p[0] for p in panels], axes.flat))

    # Track first x-value where mean crosses solved, per condition per reward panel.
    # Used to draw a vertical "first solved" marker after all lines are plotted.
    _reward_cols = {"actor_extrinsic_reward", "learner_eval_reward"}
    first_solved_x: dict[str, dict[str, float]] = {}

    for idx, (condition, csv_list) in enumerate(paths.items()):
        color = _PALETTE[idx % len(_PALETTE)]
        ls    = _LINESTYLES[idx % len(_LINESTYLES)]
        first_solved_x[condition] = {}
        for col, _ylabel, _subtitle, do_smooth, _higher in panels:
            ax = ax_map[col]
            x, mean, std = _aggregate(csv_list, col, smooth=do_smooth, x_col=x_col)
            ax.plot(x, mean, color=color, linestyle=ls, linewidth=1.8, label=condition)
            ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
            if solved is not None and col in _reward_cols:
                crossed = np.where(mean >= solved)[0]
                if len(crossed):
                    first_solved_x[condition][col] = float(x[crossed[0]])

    for col, ylabel, subtitle, _, higher_is_better in panels:
        ax = ax_map[col]
        ax.set_title(subtitle, fontsize=9, loc="left")
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.legend(fontsize=8, loc="best")
        ax.tick_params(labelsize=8)

        if col == "learner_loss":
            ax.set_yscale("log")

        if higher_is_better is True:
            ax.annotate("▲ better", xy=(0.01, 0.97), xycoords="axes fraction",
                        fontsize=7, color="green", va="top")
        elif higher_is_better is False:
            ax.annotate("▼ better", xy=(0.01, 0.97), xycoords="axes fraction",
                        fontsize=7, color="green", va="top")

        if solved is not None and col in _reward_cols:
            # Horizontal solved threshold line
            ax.axhline(solved, color="black", linewidth=1.0, linestyle=":", alpha=0.6)
            ax.annotate(f"solved ({solved:g})", xy=(0.01, solved), xycoords=("axes fraction", "data"),
                        fontsize=7, color="black", alpha=0.7, va="bottom")
            # Vertical "first solved" markers per condition
            for idx, (condition, col_map) in enumerate(first_solved_x.items()):
                if col in col_map:
                    color = _PALETTE[idx % len(_PALETTE)]
                    ax.axvline(col_map[col], color=color, linestyle=":", linewidth=1.0, alpha=0.7)
            # Cap y-axis just above solved so outliers don't compress the interesting range
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, min(hi, solved * 1.15) if solved > 0 else max(hi, solved * 1.15))

    fig.tight_layout()
    dest = out_dir or Path(_RUNS_DIR)
    out_path = dest / "comparison.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {out_path}")

    if show:
        plt.show()

    plt.close(fig)
    return out_path


def main() -> None:
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 7])
    parser.add_argument("--experiment", default="eder_vs_baseline",
                        choices=list(EXPERIMENTS),
                        help="Which experiment suite to run")
    parser.add_argument("--env", default="cartpole",
                        choices=list(ENV_PRESETS),
                        help="Environment preset")
    parser.add_argument("--workers", type=int, default=None,
                        help="Max parallel processes (default: one per job)")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training, just regenerate the plot from existing runs")
    parser.add_argument("--force", action="store_true",
                        help="Delete and re-run any existing runs for this experiment. "
                             "Without --force, runs whose config hash already exists are skipped.")
    parser.add_argument("--x-axis", choices=["episode", "env_steps"], default=None,
                        help="X-axis: episode (default) or env_steps. "
                             "sample_efficiency experiment defaults to env_steps.")
    args = parser.parse_args()

    if args.plot_only:
        paths = _build_conditions(args.experiment, args.env, args.seeds)
    else:
        paths = run_all(args.experiment, args.env, args.seeds, max_workers=args.workers, force=args.force)

    out_dir = _compare_dir(args.seeds, args.experiment, args.env, paths)

    manifest = {
        "experiment": args.experiment,
        "env": args.env,
        "seeds": args.seeds,
        "conditions": {k: [str(p) for p in v] for k, v in paths.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # Default x-axis: env_steps for sample_efficiency, episode otherwise
    if args.x_axis is not None:
        x_col = "total_env_steps" if args.x_axis == "env_steps" else "episode"
    else:
        x_col = "total_env_steps" if args.experiment == "sample_efficiency" else "episode"

    title = f"{args.experiment} | {args.env} — mean ± std across seeds {args.seeds}"
    compare(paths, out_dir=out_dir, show=args.show, title=title, x_col=x_col)


if __name__ == "__main__":
    main()

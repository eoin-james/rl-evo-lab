"""Experiment definition and runner.

An experiment is a set of algorithm conditions compared across multiple seeds.
Define one in a script, run it directly:

    python experiments/cartpole_efficiency.py
    python experiments/cartpole_efficiency.py --force --show

Or from Python (e.g. a notebook)::

    from experiments.cartpole_efficiency import experiment

    experiment.run()
    experiment.run(force=True, seeds=[42])        # quick single-seed re-run
    experiment.run_one("EDER", seed=42)           # one condition, one seed
    experiment.plot(show=True)                    # re-plot without re-training
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import queue
import shutil
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.live import Live
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from rl_evo_lab.train import train
from rl_evo_lab.utils.config import EDERConfig, make_config
from rl_evo_lab.utils.logging import _run_id

_console = Console()
_RUNS_DIR = "runs"


# ---------------------------------------------------------------------------
# Condition
# ---------------------------------------------------------------------------

class Condition:
    """A labelled set of config overrides defining one experimental condition.

    Pass any EDERConfig field as a keyword argument::

        Condition("EDER",   use_es=True,  use_novelty=True)
        Condition("ES+DQN", use_es=True,  use_novelty=False)
        Condition("DQN",    use_es=False, use_novelty=False)
        Condition("Big",    hidden_dim=256)
    """

    def __init__(self, label: str, **overrides: Any) -> None:
        self.label = label
        self.overrides: dict[str, Any] = overrides

    def __repr__(self) -> str:
        kv = ", ".join(f"{k}={v!r}" for k, v in self.overrides.items())
        return f"Condition({self.label!r}, {kv})"


# ---------------------------------------------------------------------------
# Status helpers  (inspired by rl-core RunManager)
# ---------------------------------------------------------------------------

def _is_done(run_dir: Path) -> bool:
    """Return True if this run completed successfully.

    Checks status.json first (written by RunLogger). Falls back to
    metrics.csv existence for runs predating status tracking.
    """
    status_path = run_dir / "status.json"
    if status_path.exists():
        return json.loads(status_path.read_text()).get("status") == "completed"
    return (run_dir / "metrics.csv").exists()


# ---------------------------------------------------------------------------
# Picklable train worker (must be module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _train_worker(args: tuple) -> tuple[str, Path]:
    cfg, q = args
    run_dir = Path(_RUNS_DIR) / _run_id(cfg)
    if not _is_done(run_dir):
        train(cfg, log_dir=_RUNS_DIR, verbose=False, progress_queue=q)
    return _run_id(cfg), run_dir / "metrics.csv"


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    """Self-contained experiment: env × conditions × seeds.

    Example::

        from rl_evo_lab.experiment import Condition, Experiment

        experiment = Experiment(
            name="cartpole_efficiency",
            env="cartpole",
            seeds=[7, 42, 123],
            conditions=[
                Condition("EDER",   use_es=True,  use_novelty=True),
                Condition("ES+DQN", use_es=True,  use_novelty=False),
                Condition("DQN",    use_es=False, use_novelty=False),
            ],
        )

        experiment.run()
    """

    name: str
    env: str
    seeds: list[int]
    conditions: list[Condition]
    # Per-experiment HP overrides applied on top of the env preset.
    # Use to make an experiment tractable (e.g. fewer episodes for a smoke test)
    # without editing the global ENV_PRESETS.
    env_overrides: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        force: bool = False,
        workers: int | None = None,
        show: bool = False,
        x_axis: str = "episode",
        results_dir: str = _RUNS_DIR,
    ) -> Path:
        """Run all conditions and seeds, then save a comparison plot.

        Already-completed runs are skipped unless ``force=True``.
        Returns the path to the saved plot PNG.
        """
        if force:
            self._delete_runs(results_dir)

        for i, cond in enumerate(self.conditions, 1):
            pending = [
                cfg for cfg in self._cfgs(cond)
                if not _is_done(Path(results_dir) / _run_id(cfg))
            ]

            _console.rule(f"[bold]{cond.label}[/bold]  ({i}/{len(self.conditions)})")

            if not pending:
                _console.print(f"  [dim]{cond.label}[/dim] — all seeds cached\n")
                continue

            self._run_condition(cond.label, pending, workers, results_dir)

        return self._make_plot(show=show, x_axis=x_axis, results_dir=results_dir)

    def plot(
        self,
        show: bool = False,
        x_axis: str = "episode",
        results_dir: str = _RUNS_DIR,
    ) -> Path:
        """Re-plot from existing runs without re-training."""
        return self._make_plot(show=show, x_axis=x_axis, results_dir=results_dir)

    def run_one(
        self,
        label: str,
        seed: int,
        force: bool = False,
        results_dir: str = _RUNS_DIR,
    ) -> Path:
        """Train a single condition + seed. Returns the metrics CSV path.

        Useful for quick iteration or debugging a specific condition::

            experiment.run_one("EDER", seed=42)
        """
        cond = self._condition(label)
        cfg = self._make_cfg(cond, seed)
        run_dir = Path(results_dir) / _run_id(cfg)

        if force and run_dir.exists():
            shutil.rmtree(run_dir)

        if _is_done(run_dir):
            _console.print(f"[dim]{label} seed={seed} — cached at {run_dir}[/dim]")
        else:
            train(cfg, log_dir=results_dir, verbose=True)

        return run_dir / "metrics.csv"

    def main(self) -> None:
        """Standard CLI entry point for experiment scripts.

        Usage in an experiment file::

            if __name__ == "__main__":
                experiment.main()

        Flags: --force, --show, --workers N, --x-axis episode|env_steps, --plot-only
        """
        import argparse
        p = argparse.ArgumentParser(description=f"Run experiment: {self.name} on {self.env}")
        p.add_argument("--force",     action="store_true", help="Delete and re-run existing results")
        p.add_argument("--show",      action="store_true", help="Open the plot after saving")
        p.add_argument("--workers",   type=int, default=None, help="Max parallel processes")
        p.add_argument("--x-axis",    choices=["episode", "env_steps"], default="episode")
        p.add_argument("--plot-only", action="store_true", help="Re-plot without re-training")
        args = p.parse_args()

        if args.plot_only:
            self.plot(show=args.show, x_axis=args.x_axis)
        else:
            self.run(force=args.force, show=args.show, workers=args.workers, x_axis=args.x_axis)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_cfg(self, cond: Condition, seed: int) -> EDERConfig:
        return make_config(self.env, seed=seed, **{**self.env_overrides, **cond.overrides})

    def _cfgs(self, cond: Condition) -> list[EDERConfig]:
        return [self._make_cfg(cond, s) for s in self.seeds]

    def _condition(self, label: str) -> Condition:
        for c in self.conditions:
            if c.label == label:
                return c
        raise ValueError(f"Unknown condition {label!r}. Available: {[c.label for c in self.conditions]}")

    def _paths(self, results_dir: str = _RUNS_DIR) -> dict[str, list[Path]]:
        return {
            cond.label: [
                Path(results_dir) / _run_id(self._make_cfg(cond, s)) / "metrics.csv"
                for s in self.seeds
            ]
            for cond in self.conditions
        }

    def _out_dir(self, paths: dict[str, list[Path]], results_dir: str = _RUNS_DIR) -> Path:
        all_ids = sorted(str(p.parent.name) for csvs in paths.values() for p in csvs)
        digest = hashlib.sha1("\n".join(all_ids).encode()).hexdigest()[:8]
        seed_key = "_".join(str(s) for s in sorted(self.seeds))
        d = Path(results_dir) / f"compare__{self.name}__{self.env}__seeds_{seed_key}__{digest}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _delete_runs(self, results_dir: str) -> None:
        for cond in self.conditions:
            for cfg in self._cfgs(cond):
                run_dir = Path(results_dir) / _run_id(cfg)
                if run_dir.exists():
                    shutil.rmtree(run_dir)

    def _run_condition(
        self,
        label: str,
        pending: list[EDERConfig],
        workers: int | None,
        results_dir: str,
    ) -> None:
        n_workers = min(len(pending), workers or len(pending))

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
        t0 = time.monotonic()

        with Live(progress, refresh_per_second=10, console=_console):
            listener.start()
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_train_worker, (cfg, q)): cfg for cfg in pending}
                for future in as_completed(futures):
                    cfg = futures[future]
                    run_id, _ = future.result()
                    if run_id in task_ids:
                        progress.update(task_ids[run_id], completed=cfg.total_episodes)
            q.put(None)
            listener.join()
        manager.shutdown()

        elapsed = time.monotonic() - t0
        mins, secs = divmod(int(elapsed), 60)
        _console.print(f"  [bold green]✓[/bold green] {label} — {mins}m {secs:02d}s\n")

    def _make_plot(self, show: bool, x_axis: str, results_dir: str) -> Path:
        from rl_evo_lab.utils.compare import compare  # avoid circular import at module level

        paths = self._paths(results_dir)
        out_dir = self._out_dir(paths, results_dir)
        manifest = {
            "experiment": self.name,
            "env": self.env,
            "seeds": self.seeds,
            "conditions": {k: [str(p) for p in v] for k, v in paths.items()},
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        x_col = "total_env_steps" if x_axis == "env_steps" else "episode"
        title = f"{self.name} | {self.env} — mean ± std across seeds {self.seeds}"
        return compare(paths, out_dir=out_dir, show=show, title=title, x_col=x_col)

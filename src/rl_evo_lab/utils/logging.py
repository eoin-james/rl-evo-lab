from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass, fields
from multiprocessing import Queue
from pathlib import Path
from typing import Any

from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeRemainingColumn

from rl_evo_lab.utils.config import EDERConfig


def _run_key(cfg: EDERConfig) -> str:
    """Stable run key: seed{N}__{8-char hash of all HPs}."""
    hp = {f.name: getattr(cfg, f.name) for f in fields(cfg) if f.name not in ("use_wandb", "wandb_project")}
    digest = hashlib.sha1(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:8]
    return f"seed{cfg.seed}__{digest}"


def _run_dir(cfg: EDERConfig, results_dir: str | Path = "runs") -> Path:
    """Canonical run directory: {results_dir}/{env_id}/seed{N}__{hash}/"""
    return Path(results_dir) / cfg.env_id / _run_key(cfg)


@dataclass
class EpisodeLog:
    episode: int
    total_env_steps: int       # cumulative env steps across all workers/episodes so far
    actor_augmented_reward: float
    actor_extrinsic_reward: float
    learner_loss: float
    learner_eval_reward: float | None
    buffer_diversity: float | None
    idn_loss: float
    effective_beta: float
    buffer_size: int
    sync: bool = False


class RunLogger:
    def __init__(
        self,
        cfg: EDERConfig,
        log_dir: str = "runs",
        verbose: bool = True,
        progress_queue: Queue | None = None,
    ) -> None:
        self.cfg = cfg
        run_dir = _run_dir(cfg, log_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = _run_key(cfg)  # kept for progress queue messages

        self._csv_path = run_dir / "metrics.csv"
        self._csv_file = self._csv_path.open("a", newline="")
        self._writer: csv.DictWriter | None = None
        self._write_header = self._csv_path.stat().st_size == 0

        self._run_dir = run_dir
        self._write_config(run_dir)
        self._write_status("running")
        self._wandb = self._init_wandb() if cfg.use_wandb else None

        # If a queue is provided, send progress updates to the parent process.
        # Otherwise show a local progress bar when verbose=True.
        self._queue: Queue | None = progress_queue
        self._last_eval: float | None = None
        self._progress: Progress | None = None
        self._task_id = None
        if verbose and progress_queue is None:
            mode = "EDER" if cfg.use_es and cfg.use_novelty else ("ES+DQN" if cfg.use_es else "DQN")
            desc = f"[cyan]{mode}[/cyan] seed={cfg.seed}"
            self._progress = Progress(
                TextColumn("{task.description}"),
                BarColumn(),
                MofNCompleteColumn(),
                TimeRemainingColumn(),
                TextColumn("{task.fields[stats]}"),
            )
            self._task_id = self._progress.add_task(desc, total=cfg.total_episodes, stats="")
            self._progress.start()

    def _write_config(self, run_dir: Path) -> None:
        cfg_path = run_dir / "config.json"
        if not cfg_path.exists():
            cfg_path.write_text(
                json.dumps({f.name: getattr(self.cfg, f.name) for f in fields(self.cfg)}, indent=2)
            )

    def _write_status(self, status: str) -> None:
        (self._run_dir / "status.json").write_text(json.dumps({"status": status}))

    def _init_wandb(self):
        try:
            import wandb
            return wandb.init(
                project=self.cfg.wandb_project,
                name=self.run_id,
                id=self.run_id,
                resume="allow",
                config={f.name: getattr(self.cfg, f.name) for f in fields(self.cfg)},
            )
        except Exception as e:
            print(f"[warn] wandb init failed: {e}")
            return None

    def log(self, entry: EpisodeLog) -> None:
        row = asdict(entry)

        # CSV
        if self._writer is None:
            self._writer = csv.DictWriter(self._csv_file, fieldnames=row.keys())
            if self._write_header:
                self._writer.writeheader()
        self._writer.writerow(row)
        self._csv_file.flush()

        # wandb
        if self._wandb is not None:
            import wandb
            payload = {k: v for k, v in row.items() if v is not None and k != "episode"}
            wandb.log(payload, step=entry.episode)

        # Progress reporting
        if entry.learner_eval_reward is not None:
            self._last_eval = entry.learner_eval_reward

        if self._queue is not None:
            msg: dict[str, Any] = {
                "run_id": self.run_id,
                "episode": entry.episode,
                "loss": entry.learner_loss,
                "buf": entry.buffer_size,
                "beta": entry.effective_beta if entry.effective_beta > 0 else None,
                "eval": self._last_eval,
                "sync": entry.sync,
            }
            self._queue.put(msg)
        elif self._progress is not None:
            parts = [f"loss={entry.learner_loss:.4f}", f"buf={entry.buffer_size:,}"]
            if entry.effective_beta > 0:
                parts.append(f"β={entry.effective_beta:.4f}")
            if self._last_eval is not None:
                parts.append(f"[green]eval={self._last_eval:.1f}[/green]")
            if entry.sync:
                parts.append("[yellow]sync[/yellow]")
            self._progress.update(self._task_id, advance=1, stats="  ".join(parts))

    def close(self) -> None:
        if self._progress is not None:
            self._progress.stop()
        self._csv_file.close()
        if self._wandb is not None:
            self._wandb.finish()
        self._write_status("completed")

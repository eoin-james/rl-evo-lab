from pathlib import Path
from rl_evo_lab.utils.config import EDERConfig
from rl_evo_lab.train import train


def test_short_run(tmp_path: Path):
    cfg = EDERConfig(
        total_episodes=3,
        es_n_workers=4,
        min_buffer_size=10,
        eval_freq=2,
        eval_episodes=1,
        seed=0,
    )
    train(cfg, log_dir=str(tmp_path))
    # structure: {log_dir}/{env_id}/seed{N}__{hash}/metrics.csv
    env_dir = next(tmp_path.iterdir())
    run_dir = next(env_dir.iterdir())
    assert (run_dir / "metrics.csv").exists()

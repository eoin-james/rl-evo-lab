from __future__ import annotations

import numpy as np
import pytest
import gymnasium as gym
import torch

from rl_evo_lab.actor.es_actor import ESActor, _rank_normalize
from rl_evo_lab.actor.es_worker import run_worker_episode
from rl_evo_lab.buffer.replay_buffer import ReplayBuffer
from rl_evo_lab.intrinsic.episodic_novelty import EpisodicNovelty
from rl_evo_lab.intrinsic.inverse_dynamics import InverseDynamicsNetwork
from rl_evo_lab.learner.network import QNetwork
from rl_evo_lab.utils.config import EDERConfig


# ---------------------------------------------------------------------------
# 1. Rank normalisation
# ---------------------------------------------------------------------------

def test_rank_normalization():
    fitnesses = np.array([10.0, 5.0, 1.0, 8.0], dtype=np.float32)
    ranks = _rank_normalize(fitnesses)

    # Min must be -0.5, max must be +0.5
    assert pytest.approx(ranks.min(), abs=1e-6) == -0.5
    assert pytest.approx(ranks.max(), abs=1e-6) == 0.5

    # The rank order must match the fitness order:
    # fitness sorted ascending: [1, 5, 8, 10] → indices [2, 1, 3, 0]
    # so ranks[2] < ranks[1] < ranks[3] < ranks[0]
    assert ranks[2] < ranks[1] < ranks[3] < ranks[0]


# ---------------------------------------------------------------------------
# 2. Worker episode produces valid WorkerResult
# ---------------------------------------------------------------------------

def test_worker_episode_returns_transitions():
    cfg = EDERConfig(es_n_workers=2)
    device = torch.device("cpu")
    env = gym.make("CartPole-v1")
    idn = InverseDynamicsNetwork(cfg, device)
    novelty = EpisodicNovelty(cfg.knn_k)

    # Build base params from a fresh network matching the config's hidden_dim
    net = QNetwork(cfg.obs_dim, cfg.act_dim, hidden=cfg.hidden_dim)
    base_params = net.get_flat_params()

    result = run_worker_episode(
        base_params=base_params,
        noise_seed=42,
        sigma=cfg.es_sigma,
        env=env,
        cfg=cfg,
        idn=idn,
        novelty=novelty,
        effective_beta=0.0,  # augmented == extrinsic when beta=0
        noise_sign=+1,
        device=device,
    )
    env.close()

    # Non-empty episode
    assert len(result.transitions) > 0

    # fitness is a plain Python float (or numpy scalar that behaves like one)
    assert isinstance(result.fitness, float)

    # With beta=0, augmented and extrinsic returns must be equal
    assert pytest.approx(result.fitness, abs=1e-5) == result.extrinsic_return

    # Noise vector must match the parameter count
    assert result.noise_vector.shape == base_params.shape

    # Transitions have the expected 5-tuple structure
    obs, action, reward, next_obs, done = result.transitions[0]
    assert obs.shape == (cfg.obs_dim,)
    assert next_obs.shape == (cfg.obs_dim,)
    assert isinstance(action, int)
    assert isinstance(reward, float)
    assert isinstance(done, bool)


# ---------------------------------------------------------------------------
# 3. ESActor updates theta_base after one generation
# ---------------------------------------------------------------------------

def test_es_actor_updates_params():
    cfg = EDERConfig(
        es_n_workers=4,
        es_antithetic=True,  # 2 seed pairs → 4 workers
        idn_updates_per_episode=1,
    )
    device = torch.device("cpu")

    actor = ESActor(cfg, device)
    idn = InverseDynamicsNetwork(cfg, device)
    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.obs_dim)

    initial_params = actor.get_base_params().copy()

    env_fn = lambda: gym.make("CartPole-v1")
    stats = actor.run_generation(env_fn, idn, buffer, episode_num=0)

    updated_params = actor.get_base_params()

    # Parameters must have changed (ES update is virtually guaranteed to move theta)
    assert not np.allclose(initial_params, updated_params), (
        "theta_base did not change after one ES generation"
    )

    # Buffer must have received transitions
    assert len(buffer) > 0

    # Stats have plausible values
    assert isinstance(stats.mean_augmented_fitness, float)
    assert isinstance(stats.mean_extrinsic_return, float)
    assert isinstance(stats.idn_loss, float)

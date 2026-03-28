from __future__ import annotations

from multiprocessing import Queue

import gymnasium as gym
import torch

from rl_evo_lab.actor.es_actor import ESActor
from rl_evo_lab.buffer.replay_buffer import ReplayBuffer
from rl_evo_lab.intrinsic.inverse_dynamics import InverseDynamicsNetwork
from rl_evo_lab.learner.dqn import DQNLearner
from rl_evo_lab.utils.config import EDERConfig
from pathlib import Path

from rl_evo_lab.utils.logging import EpisodeLog, RunLogger
from rl_evo_lab.utils.seeding import seed_everything


def train(
    cfg: EDERConfig = EDERConfig(),
    log_dir: str = "runs",
    verbose: bool = True,
    progress_queue: Queue | None = None,
    run_dir: Path | None = None,
) -> None:
    seed_everything(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    buffer = ReplayBuffer(cfg.buffer_capacity, cfg.obs_dim)
    learner = DQNLearner(cfg, device)
    idn = InverseDynamicsNetwork(cfg, device) if cfg.use_es else None
    actor = ESActor(cfg, device) if cfg.use_es else None
    logger = RunLogger(cfg, log_dir=log_dir, verbose=verbose, progress_queue=progress_queue, run_dir=run_dir)

    env_fn = lambda: gym.make(cfg.env_id)
    collect_env = gym.make(cfg.env_id)  # used by DQN collect_episode when use_es=False
    eval_env = gym.make(cfg.env_id)

    last_loss = 0.0
    last_eval = 0.0
    cumulative_env_steps = 0

    # Early stopping state
    _solved_streak = 0          # consecutive eval windows at or above solved_reward
    _best_eval = -float("inf")  # best eval seen so far
    _stale_count = 0            # consecutive eval windows without meaningful improvement

    for episode in range(cfg.total_episodes):
        eval_reward = None
        diversity = None
        did_sync = False
        idn_loss = 0.0

        if cfg.use_es:
            stats = actor.run_generation(env_fn, idn, buffer, episode)
            mean_extrinsic_return = stats.mean_extrinsic_return
            mean_augmented_fitness = stats.mean_augmented_fitness
            idn_loss = stats.idn_loss
            effective_beta = stats.effective_beta
            cumulative_env_steps += stats.total_env_steps
        else:
            ep_return, ep_steps = learner.collect_episode(collect_env, buffer, episode)
            mean_extrinsic_return = ep_return
            mean_augmented_fitness = ep_return
            effective_beta = 0.0
            cumulative_env_steps += ep_steps

        if len(buffer) >= cfg.min_buffer_size:
            for _ in range(cfg.learner_updates_per_episode):
                last_loss = learner.train_step(buffer)

        # Evaluate at eval_freq for all conditions — keeps chart resolution consistent.
        if episode % cfg.eval_freq == 0:
            eval_reward = learner.evaluate(eval_env, cfg.eval_episodes)
            last_eval = eval_reward
            diversity = buffer.diversity_metric()
            if cfg.use_es:
                actor.update_learner_eval(last_eval)
                # Sync immediately when solved: anchors ES to the working policy
                # every eval cycle, cutting off the forgetting cycle at its root.
                if last_eval >= cfg.solved_reward:
                    actor.sync_from_learner(learner.get_weights())
                    did_sync = True

            # Early stopping trackers — updated every eval window
            if last_eval >= cfg.solved_reward:
                _solved_streak += 1
            else:
                _solved_streak = 0

            if last_eval > _best_eval + cfg.early_stop_min_delta:
                _best_eval = last_eval
                _stale_count = 0
            else:
                _stale_count += 1

        # Periodic pre-solve sync: pulls ES toward learner before it fully solves,
        # preventing the actor from diverging too far. Skipped when solved (handled above).
        if cfg.use_es and not did_sync and episode % cfg.sync_freq == 0:
            threshold = cfg.sync_eval_threshold * mean_extrinsic_return
            if last_eval >= threshold:
                actor.sync_from_learner(learner.get_weights())
                did_sync = True

        logger.log(EpisodeLog(
            episode=episode,
            total_env_steps=cumulative_env_steps,
            actor_augmented_reward=mean_augmented_fitness,
            actor_extrinsic_reward=mean_extrinsic_return,
            learner_loss=last_loss,
            learner_eval_reward=eval_reward,
            buffer_diversity=diversity,
            idn_loss=idn_loss,
            effective_beta=effective_beta,
            buffer_size=len(buffer),
            sync=did_sync,
        ))

        # Check early stopping after logging so the final episode is always in the CSV
        if episode % cfg.eval_freq == 0:
            if _solved_streak >= cfg.early_stop_solved_window:
                break
            if _stale_count >= cfg.early_stop_patience:
                break

    logger.close()
    collect_env.close()
    eval_env.close()


if __name__ == "__main__":
    train()

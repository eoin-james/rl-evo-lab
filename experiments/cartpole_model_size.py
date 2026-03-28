"""Does ES compensate for a smaller network?

Tests whether the diversity injected by ES allows a smaller network to
match or exceed a larger pure-DQN network.

Run:
    python experiments/cartpole_model_size.py
"""
from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="cartpole_model_size",
    env="cartpole",
    seeds=[7, 42, 123],
    conditions=[
        Condition("EDER-64",  use_es=True,  use_novelty=True,  hidden_dim=64),
        Condition("EDER-128", use_es=True,  use_novelty=True,  hidden_dim=128),
        Condition("DQN-64",   use_es=False, use_novelty=False, hidden_dim=64),
        Condition("DQN-128",  use_es=False, use_novelty=False, hidden_dim=128),
    ],
)

if __name__ == "__main__":
    experiment.main()

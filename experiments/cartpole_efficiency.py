"""Does the ES actor improve sample efficiency vs pure DQN?

Three conditions on CartPole-v1:
  EDER    — ES actor + DQN learner + IDN novelty
  ES+DQN  — ES actor + DQN learner, no novelty
  DQN     — pure DQN with ε-greedy, no ES

Run:
    python experiments/cartpole_efficiency.py
    python experiments/cartpole_efficiency.py --force --show
    python experiments/cartpole_efficiency.py --plot-only --show
"""
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

if __name__ == "__main__":
    experiment.main()

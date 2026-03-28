"""Does the ES actor improve sample efficiency vs pure DQN on LunarLander?

LunarLander-v3 is a better benchmark for EDER than CartPole:
  - Longer episodes (~400 steps) give IDN more signal to learn from
  - Partial rewards (leg contact, proximity to pad) provide training signal
  - Diverse failure modes reward exploration-driven variety in the buffer

Run:
    python experiments/lunarlander_efficiency.py
    python experiments/lunarlander_efficiency.py --force --show
"""
from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="lunarlander_efficiency",
    env="lunarlander",
    seeds=[7, 42, 123],
    conditions=[
        Condition("EDER",   use_es=True,  use_novelty=True),
        Condition("ES+DQN", use_es=True,  use_novelty=False),
        Condition("DQN",    use_es=False, use_novelty=False),
    ],
)

if __name__ == "__main__":
    experiment.main()

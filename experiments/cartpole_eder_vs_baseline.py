"""Does IDN novelty help the ES actor? EDER vs ES+DQN baseline.

Both conditions use ES + DQN. The only difference is whether the IDN
novelty signal is added to the actor's fitness.

Run:
    python experiments/cartpole_eder_vs_baseline.py
"""
from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="cartpole_eder_vs_baseline",
    env="cartpole",
    seeds=[7, 42, 123],
    conditions=[
        Condition("EDER",     use_es=True, use_novelty=True),
        Condition("Baseline", use_es=True, use_novelty=False),
    ],
)

if __name__ == "__main__":
    experiment.main()

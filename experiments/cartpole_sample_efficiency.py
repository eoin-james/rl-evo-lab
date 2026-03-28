"""Fair env-step budget comparison: ES vs DQN.

ES uses N workers per episode, so it consumes N× more env steps than DQN
per training iteration. This experiment gives DQN proportionally more
episodes so all conditions have the same total env-step budget.

Use --x-axis env_steps to make the fair comparison visible in the plot.

Run:
    python experiments/cartpole_sample_efficiency.py --x-axis env_steps
"""
from rl_evo_lab.experiment import Condition, Experiment

experiment = Experiment(
    name="cartpole_sample_efficiency",
    env="cartpole",
    seeds=[7, 42, 123],
    conditions=[
        Condition("EDER",   use_es=True,  use_novelty=True),
        Condition("ES+DQN", use_es=True,  use_novelty=False),
        # DQN gets more episodes to match the ES env-step budget
        Condition("DQN",    use_es=False, use_novelty=False, total_episodes=10_000),
    ],
)

if __name__ == "__main__":
    import sys
    # Default to env_steps x-axis for this experiment
    if "--x-axis" not in sys.argv:
        sys.argv += ["--x-axis", "env_steps"]
    experiment.main()

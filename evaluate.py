"""Compare fixed / random / Q-learning on every jammer. Run: python evaluate.py"""
import csv
import os
import random
import statistics

from simulator import CommunicationSimulator
from train import train_agent, NUM_CHANNELS, RESULTS_DIR

EVAL_TRIALS = 30
EVAL_STEPS = 1000
STRATEGIES = ("fixed", "random", "q_learning")


def evaluate_strategy(strategy, jammer_type, agent=None, seed=1234):
    rng = random.Random(seed)
    trial_pdrs = []
    for t in range(EVAL_TRIALS):
        env = CommunicationSimulator(NUM_CHANNELS, jammer_type, seed=seed + t)
        env.reset()
        ok = 0
        for _ in range(EVAL_STEPS):
            if strategy == "fixed":
                ch = 0
            elif strategy == "random":
                ch = rng.randrange(NUM_CHANNELS)
            elif strategy == "q_learning":
                ch = agent.choose_action(env.get_state())  # exploration = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            _, success = env.step(ch)
            ok += int(success)
        trial_pdrs.append(100 * ok / EVAL_STEPS)
    return statistics.mean(trial_pdrs), statistics.pstdev(trial_pdrs)


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    rows = []
    for jammer in CommunicationSimulator.JAMMER_TYPES:
        print(f"\n--- Jammer: {jammer} ---")
        agent = train_agent(jammer)
        agent.save(os.path.join(RESULTS_DIR, f"q_table_{jammer}.json"))
        for s in STRATEGIES:
            mean, sd = evaluate_strategy(s, jammer, agent)
            print(f"{s:12s} PDR: {mean:6.2f}%  (std {sd:.2f})")
            rows.append([jammer, s, round(mean, 2), round(sd, 2)])
    with open(os.path.join(RESULTS_DIR, "evaluation_results.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["jammer", "strategy", "pdr_percent", "std"])
        w.writerows(rows)
    print("\nSaved results/evaluation_results.csv and results/q_table_*.json")

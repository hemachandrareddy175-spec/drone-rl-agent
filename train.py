"""Train one agent per jammer and save Q-tables to ../results/."""
import os
from simulator import CommunicationSimulator
from q_learning import QLearningAgent

NUM_CHANNELS = 8
EPISODES = 500
STEPS_PER_EPISODE = 100
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def train_agent(jammer_type="sweep", seed=0, verbose=False):
    agent = QLearningAgent(num_channels=NUM_CHANNELS, learning_rate=0.1,
                           discount_factor=0.9, exploration_rate=0.2, seed=seed)
    env = CommunicationSimulator(NUM_CHANNELS, jammer_type, seed=seed)
    ok = total = 0
    for ep in range(EPISODES):
        # exploration decays 0.2 -> 0.02 so training converges cleanly
        agent.exploration_rate = max(0.02, 0.2 * (1 - ep / (0.8 * EPISODES)))
        env.reset()
        state = env.get_state()
        for _ in range(STEPS_PER_EPISODE):
            action = agent.choose_action(state)
            reward, success = env.step(action)
            next_state = env.get_state()
            agent.learn(state, action, reward, next_state)
            state = next_state
            total += 1
            ok += int(success)
        if verbose and (ep + 1) % 100 == 0:
            print(f"  [{jammer_type}] episode {ep+1}/{EPISODES} cumulative training PDR {100*ok/total:.2f}%")
    agent.exploration_rate = 0.0
    return agent


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for j in CommunicationSimulator.JAMMER_TYPES:
        print(f"Training vs {j} jammer")
        a = train_agent(j, verbose=True)
        path = os.path.join(RESULTS_DIR, f"q_table_{j}.json")
        a.save(path)
        print(f"  saved {os.path.normpath(path)}")

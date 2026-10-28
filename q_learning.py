"""Tabular Q-learning agent for channel selection (states x channels)."""
import json
import random


class QLearningAgent:
    def __init__(self, num_channels=8, num_states=None, learning_rate=0.1,
                 discount_factor=0.9, exploration_rate=0.2, seed=None):
        self.num_channels = num_channels
        self.num_states = num_states if num_states is not None else num_channels
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.rng = random.Random(seed)
        self.q_values = [[0.0] * num_channels for _ in range(self.num_states)]

    def choose_action(self, state):
        if self.rng.random() < self.exploration_rate:
            return self.rng.randrange(self.num_channels)
        row = self.q_values[state]
        best = max(row)
        return self.rng.choice([c for c, v in enumerate(row) if v == best])

    def learn(self, state, action, reward, next_state):
        old = self.q_values[state][action]
        target = reward + self.discount_factor * max(self.q_values[next_state])
        self.q_values[state][action] = old + self.learning_rate * (target - old)

    # --- persistence, for integrating into your main simulation ---
    def save(self, path):
        with open(path, "w") as f:
            json.dump({"num_channels": self.num_channels,
                       "num_states": self.num_states,
                       "q_values": self.q_values}, f, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        agent = cls(num_channels=d["num_channels"], num_states=d["num_states"],
                    exploration_rate=0.0)
        agent.q_values = d["q_values"]
        return agent

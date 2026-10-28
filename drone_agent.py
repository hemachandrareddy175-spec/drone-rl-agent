"""Drop-in module for your main drone simulation.

    from drone_agent import ChannelSelector
    sel = ChannelSelector("sweep")            # trains (or loads) automatically
    ch = sel.select_channel(state)            # state = jammer/link state 0..7
    sel.update(state, ch, reward, next_state) # optional online learning
"""
import os
from q_learning import QLearningAgent
from train import train_agent, RESULTS_DIR


class ChannelSelector:
    def __init__(self, jammer_type="sweep"):
        path = os.path.join(RESULTS_DIR, f"q_table_{jammer_type}.json")
        self.agent = QLearningAgent.load(path) if os.path.exists(path) else train_agent(jammer_type)
        self.agent.exploration_rate = 0.0

    def select_channel(self, state):
        return self.agent.choose_action(state)

    def update(self, state, action, reward, next_state, explore=0.0):
        self.agent.exploration_rate = explore
        self.agent.learn(state, action, reward, next_state)

"""Layer 1 communication simulator: one drone link, N channels, one jammer.

Interface (unchanged, safe to plug into your bigger simulation):
    env = CommunicationSimulator(num_channels=8, jammer_type="sweep", seed=None)
    env.reset()                -> initial state (int)
    env.get_state()            -> current state (int, 0..num_channels-1)
    env.step(channel)          -> (reward, success)   reward = +1 / -1

Jammer models (the jammer hits exactly ONE channel per step):
    random   : jams a uniformly random channel each step. Unpredictable, so the
               best any agent can do is 1 - 1/N (87.5% for N=8).
    sweep    : jams channel 0,1,2,... cyclically. State = the channel that will
               be jammed this step (assumes the drone can sense the sweep).
    reactive : jams the channel the drone used on its previous packet (with
               probability `reactive_hit_prob`). State = the drone's own last
               channel. Staying on one channel is punished; hopping wins.
"""
import random


class CommunicationSimulator:
    JAMMER_TYPES = ("random", "sweep", "reactive")

    def __init__(self, num_channels=8, jammer_type="random",
                 reactive_hit_prob=1.0, seed=None):
        if jammer_type not in self.JAMMER_TYPES:
            raise ValueError(f"Unknown jammer type: {jammer_type}")
        self.num_channels = num_channels
        self.jammer_type = jammer_type
        self.reactive_hit_prob = reactive_hit_prob
        self.rng = random.Random(seed)
        self.current_step = 0
        self.last_channel = 0
        self.sweep_offset = 0

    def reset(self):
        """Start a new episode. Sweep starts at a random phase."""
        self.current_step = 0
        self.last_channel = 0
        self.sweep_offset = self.rng.randrange(self.num_channels)
        return self.get_state()

    def get_state(self):
        if self.jammer_type == "sweep":
            return (self.current_step + self.sweep_offset) % self.num_channels
        if self.jammer_type == "reactive":
            return self.last_channel
        return 0  # random jammer: nothing to observe

    def step(self, chosen_channel):
        if not 0 <= chosen_channel < self.num_channels:
            raise ValueError("Chosen channel is outside the valid range.")

        if self.jammer_type == "random":
            jammed_channel = self.rng.randrange(self.num_channels)
        elif self.jammer_type == "sweep":
            jammed_channel = self.get_state()
        else:  # reactive
            if self.rng.random() < self.reactive_hit_prob:
                jammed_channel = self.last_channel
            else:
                jammed_channel = self.rng.randrange(self.num_channels)

        success = chosen_channel != jammed_channel
        self.last_channel = chosen_channel
        self.current_step += 1
        return (1 if success else -1), success

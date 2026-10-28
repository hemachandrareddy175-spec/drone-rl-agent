# Layer 1 RL - anti-jamming channel selection
Run (from layer1_rl/):  python train.py   |  python evaluate.py
Results: results/evaluation_results.csv, results/q_table_<jammer>.json
Integration: see layer1_rl/drone_agent.py (ChannelSelector).
Assumptions: 8 channels, jammer hits one channel per step. Sweep: agent observes
the jammer's current channel. Reactive: jammer hits the drone's previous channel.
Random jammer is unpredictable -> 87.5% is the theoretical ceiling.
Simulation only, not real hardware.

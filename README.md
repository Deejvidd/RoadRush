# Road Rush

**AI Learning Lab** — A real-time neural network training environment where you watch an AI learn to drive through procedurally-generated traffic using DQN (Deep Q-Network) reinforcement learning.

## Features

- **DQN + Experience Replay**: Double DQN with soft target updates (Polyak averaging) for stable Q-learning
- **Imitation Learning (IL)**: Pre-train the AI by recording your own driving patterns, then fine-tune with RL
- **Live Training**: Watch epsilon decay, Q-values, and policy entropy in real-time as the network learns
- **Model Racing**: Load snapshots from different training episodes and race them against each other side-by-side
- **Fast Training**: Bulk train 50–2000 episodes in seconds; snapshot and resume at any point
- **Deterministic Physics**: Fixed seed per evaluation level ensures reproducible obstacle sequences across runs

## How It Works

1. **Play manually** — Drive through traffic, rack up a score. Your run is recorded as a ghost.
2. **Imitation Learn** — Optional: load recorded demos and pre-train the network to mimic human driving.
3. **RL Training** — Turn on RL; the network explores with epsilon-greedy policy, learns from reward, and improves autonomously.
4. **Compare & Race** — Replay your ghost against the current AI, or race multiple snapshots to visualize learning progress.

## Tech Stack

- **Frontend**: Vanilla JS + Canvas (60 FPS game loop + neural network inference)
- **Backend**: Flask + NumPy (pure numpy forward/backward pass, no deep learning frameworks)
- **Network**: 30 → 256 → 256 → 3 (fully connected, ReLU hidden layers)
- **Obs**: 5×3 grid occupancy + player state + spawn timer + nearest obstacles (30 features total)

## Quick Start

```bash
pip install flask numpy
python ai_server_rl.py
# Opens http://127.0.0.1:8080

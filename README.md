# Infinite Bunner — Deep RL Agent

A deep reinforcement learning agent that learns to play **Infinite Bunner**, a procedurally generated Frogger-style game built with Pygame Zero. The project includes both DQN and PPO implementations with a shared CNN+LSTM backbone.

## Game Overview

Infinite Bunner is an endless-runner where a bunny must cross procedurally generated rows of roads (dodge cars), rivers (hop on logs), rail tracks (avoid trains), and grass/hedges (navigate gaps). The level never repeats, so the agent must learn general survival and navigation strategies rather than memorizing patterns.

## Project Structure

```
├── bunner.py          # Game engine + RL training loop (Pygame Zero)
├── dqn.py             # DQN agent (DRQN with CNN+LSTM)
├── ppo.py             # PPO agent (actor-critic with CNN+LSTM)
├── logger.py          # Training metrics logger (CSV + matplotlib plots)
├── images/            # Game sprites
├── sounds/            # Game audio
├── music/             # Background music
└── high.txt           # High score persistence
```

## Architecture

Both agents share the same convolutional backbone that processes raw screen pixels:

```
Input: (batch, seq_len, 80, 80, 1) — grayscale frames
  → Conv2D(32, 8×8, stride 4, ReLU)
  → Conv2D(64, 4×4, stride 2, ReLU)
  → Conv2D(64, 3×3, stride 1, ReLU)
  → Flatten (per frame, via TimeDistributed)
  → LSTM(256)
  → LayerNormalization
```

The LSTM processes a sequence of 4 frames, giving the agent temporal awareness for tracking moving objects like cars and logs.

**DQN head:** Single dense layer → Q-values for 5 actions.

**PPO heads:** Separate actor (policy logits) and critic (state value) branches, each with a 256-unit hidden layer.

### Action Space

| Action | Direction |
|--------|-----------|
| 0      | Up        |
| 1      | Right     |
| 2      | Down      |
| 3      | Left      |
| 4      | No-op     |

### Reward Design

| Signal                | Reward  |
|-----------------------|---------|
| Forward progress      | +1.0 per row |
| Invalid/blocked move  | −0.05   |
| Death (car/water/train) | −1.0  |

## Agents

### DQN (dqn.py)

Deep Recurrent Q-Network with experience replay and a target network.

| Hyperparameter         | Default   |
|------------------------|-----------|
| Replay buffer size     | 50,000    |
| Learning rate          | 6e-5      |
| Discount factor (γ)    | 0.93      |
| Epsilon decay          | 50,000 steps |
| Target network sync    | Every 1,000 steps |
| Batch size             | 32        |
| Sequence length        | 4 frames  |

### PPO (ppo.py)

Proximal Policy Optimization with GAE and a clipped surrogate objective.

| Hyperparameter         | Default   |
|------------------------|-----------|
| Learning rate          | 3e-4      |
| Discount factor (γ)    | 0.93      |
| GAE λ                  | 0.95      |
| Clip epsilon           | 0.2       |
| Entropy coefficient    | 0.01      |
| Value coefficient      | 0.5       |
| Horizon (rollout)      | 128 steps |
| PPO epochs per update  | 4         |
| Mini-batch size        | 32        |
| Sequence length        | 4 frames  |

## Setup

### Requirements

- Python ≥ 3.5
- Pygame Zero ≥ 1.2
- TensorFlow ≥ 2.x
- OpenCV (`cv2`)
- NumPy
- Matplotlib

### Install

```bash
pip install pgzero tensorflow opencv-python numpy matplotlib
```

## Usage

### Training with PPO (default)

In `bunner.py`, the agent is configured at the bottom of the file. To train with PPO:

```python
from ppo import PPO

ppo = PPO(
    num_actions=5,
    learning_rate=3e-4,
    discount_factor=0.95,
    horizon=128,
    ppo_epochs=4,
    mini_batch_size=32,
    observation_limit=1_000,
    save_frequency=5_000,
    seq_len=4,
)
```

Then run:

```bash
python bunner.py
```

Press **Space** on the title screen to start training. The agent takes over immediately — no human input needed during play.

### Training with DQN

Swap the import and instantiation in `bunner.py`:

```python
from dqn import DQN

dqn = DQN(
    num_actions=5,
    replay_size=50_000,
    learning_rate=6e-5,
    discount_factor=0.93,
    init_epsilon=0.9,
    final_epsilon=0.05,
    epsilon_decay=50_000,
    observation_limit=10_000,
    batch_size=32,
    target_update_frequency=1000,
    save_frequency=1_000,
    seq_len=4,
)
```

### Switching agents

Both agents expose the same interface — `train_step(state, game_step)` and `state_buffer`. After changing the import, update the three references in `bunner.py`:

- `game_step()`: change `dqn.state_buffer.push(frame)` → `ppo.state_buffer.push(frame)`
- `_reset_episode()`: change `dqn.state_buffer.reset()` → `ppo.state_buffer.reset()`
- `update()`: change `dqn.train_step(...)` → `ppo.train_step(...)`

Or simply alias: `dqn = ppo` after instantiation.

## Training Output

Checkpoints and logs are saved to:

| Agent | Checkpoints              | Logs                    |
|-------|--------------------------|-------------------------|
| DQN   | `./bunner/training/`     | `./bunner/logs/`        |
| PPO   | `./bunner/training_ppo/` | `./bunner/logs_ppo/`    |

The logger writes a `log` file with columns: Episode, Step, Epsilon, MeanReward, MeanLength, MeanLoss, MeanQValue, TimeDelta, and Time. It also generates plots for reward, episode length, loss, and Q-value trends.

### Expected Training Trajectory (PPO)

Based on observed runs:

| Episodes    | Mean Reward | Notes                              |
|-------------|-------------|-------------------------------------|
| 0–1,000     | 0.5–1.5     | Learning basic forward movement     |
| 1,000–5,000 | 1.5–2.5     | Navigating roads and water          |
| 5,000–10,000| 2.0–3.0     | Handling mixed terrain consistently |
| 50,000+     | Convergence | Expect diminishing returns          |

## License

Game assets and base game code from [Code the Classics](https://github.com/Wireframe-Magazine/Code-the-Classics) by Wireframe Magazine.

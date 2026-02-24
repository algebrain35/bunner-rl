
import tensorflow as tf
from tensorflow.keras import layers, models
import random
from collections import namedtuple, deque
import numpy as np
import os
from logger import MetricLogger

NUM_ACTIONS = 5  # 0=up, 1=right, 2=down, 3=left, 4=no-op

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class StateBuffer:
    """
    Maintains a rolling window of the last `seq_len` frames.
    Capacity should be >= seq_len; keeping it equal is fine since
    we only ever need one sequence at a time.
    """
    def __init__(self, seq_len: int = 4):
        self.seq_len = seq_len
        self.buffer = deque(maxlen=seq_len)

    def push(self, frame: np.ndarray):
        """Add a single (80, 80, 1) frame."""
        self.buffer.append(frame)

    def get_sequence(self) -> np.ndarray:
        """
        Returns (seq_len, 80, 80, 1) array.
        Pads with the oldest frame if the buffer isn't full yet.
        """
        if len(self.buffer) == 0:
            return np.zeros((self.seq_len, 80, 80, 1), dtype=np.float32)

        frames = list(self.buffer)
        if len(frames) < self.seq_len:
            pad = [frames[0]] * (self.seq_len - len(frames))
            frames = pad + frames

        return np.array(frames, dtype=np.float32)  # (seq_len, 80, 80, 1)

    def is_ready(self) -> bool:
        return len(self.buffer) >= self.seq_len

    def reset(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


def create_network(num_actions: int) -> tf.keras.Model:
    """
    DRQN: CNN feature extractor per frame → LSTM → Q-values.
    Input shape: (batch, seq_len, 80, 80, 1)
    """
    model = models.Sequential([
        layers.Input(shape=(None, 80, 80, 1)),
        layers.TimeDistributed(layers.Conv2D(32, (8, 8), strides=4, activation="relu")),
        layers.TimeDistributed(layers.Conv2D(64, (4, 4), strides=2, activation="relu")),
        layers.TimeDistributed(layers.Conv2D(64, (3, 3), strides=1, activation="relu")),
        layers.TimeDistributed(layers.Flatten()),
        layers.LSTM(256, return_sequences=False),
        layers.LayerNormalization(),
        layers.Dense(256, activation="relu"),
        layers.Dense(num_actions, activation="linear", name="q_values"),
    ])
    return model


class DQN:
    def __init__(
        self,
        num_actions: int = NUM_ACTIONS,
        replay_size: int = 30_000,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.95,
        init_epsilon: float = 1.0,
        final_epsilon: float = 0.1,
        epsilon_decay: int = 30_000,
        observation_limit: int = 10_000,
        batch_size: int = 32,
        target_update_frequency: int = 1_000,
        save_frequency: int = 1_000,
        seq_len: int = 4,
        checkpoint_dir: str = "./bunner/training",
        log_dir: str = "./bunner/logs",
    ):
        self.num_actions = num_actions
        self.discount = discount_factor
        self.eps = init_epsilon
        self.eps_final = final_epsilon
        self.eps_decay = epsilon_decay
        self.lr = learning_rate
        self.batch_size = batch_size
        self.target_update_frequency = target_update_frequency
        self.save_frequency = save_frequency
        self.observation_limit = observation_limit
        self.checkpoint_dir = checkpoint_dir

        self.observations = 0
        self.iterations = 0
        self.current_epsilon = init_epsilon

        self.replay = ReplayMemory(replay_size)
        self.state_buffer = StateBuffer(seq_len=seq_len)

        self.loss_fn = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr, clipnorm=10.0)

        self.main_network = create_network(num_actions)
        self.target_network = create_network(num_actions)
        # Sync weights at init
        self.target_network.set_weights(self.main_network.get_weights())

        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer,
            model=self.main_network,
        )
        self.checkpoint_path = os.path.join(checkpoint_dir, "dqn_train")
        self._restore_checkpoint()

        self.logger = MetricLogger(log_dir)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def _restore_checkpoint(self):
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        latest = tf.train.latest_checkpoint(self.checkpoint_dir)
        if latest:
            self.checkpoint.restore(latest)
            print(f"Restored checkpoint: {latest}")
        else:
            print("No checkpoint found — starting fresh.")

    def load_weights(self, filepath: str):
        self.main_network.load_weights(filepath)
        self.target_network.set_weights(self.main_network.get_weights())
        print(f"Loaded weights from {filepath}")

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def _current_epsilon(self) -> float:
        eps = self.eps_final + (self.eps - self.eps_final) * np.exp(
            -self.iterations / self.eps_decay
        )
        return float(max(self.eps_final, eps))

    def choose_action(self, state_seq: np.ndarray) -> int:
        """
        Epsilon-greedy action selection.
        state_seq: (1, seq_len, 80, 80, 1)
        """
        self.current_epsilon = self._current_epsilon()
        self.observations += 1

        if random.random() > self.current_epsilon:
            q_values = self.main_network(state_seq, training=False)
            return int(tf.argmax(q_values[0]).numpy())
        else:
            return random.randrange(self.num_actions)

    # ------------------------------------------------------------------
    # Training step (called once per game action)
    # ------------------------------------------------------------------

    def train_step(self, state: np.ndarray, game_step) -> np.ndarray:
        """
        1. Build current state sequence from buffer.
        2. Choose action.
        3. Step the game.
        4. Store transition.
        5. Optimise if ready.
        Returns the next state array (80, 80, 1).
        """
        # Capture sequence *before* the action
        current_seq = self.state_buffer.get_sequence()          # (seq_len, 80, 80, 1)
        actionable = np.expand_dims(current_seq, axis=0)        # (1, seq_len, 80, 80, 1)

        action = self.choose_action(actionable)

        print(
            f"ε={self.current_epsilon:.3f} | "
            f"obs={self.observations} | "
            f"action={action} | "
            f"replay={len(self.replay)}"
        )

        # game_step pushes the new frame into state_buffer internally
        next_state_arr, reward, terminal = game_step(action)

        # Capture sequence *after* the action
        next_seq = self.state_buffer.get_sequence()             # (seq_len, 80, 80, 1)

        self.replay.push(current_seq, action, next_seq, reward, terminal)

        if self.observations > self.observation_limit and len(self.replay) >= self.batch_size:
            self._optimize()

        return next_state_arr

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------

    def _optimize(self):
        samples = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*samples))

        states     = np.stack(batch.state).astype(np.float32)      # (B, seq, 80, 80, 1)
        next_states= np.stack(batch.next_state).astype(np.float32)
        actions    = np.array(batch.action,   dtype=np.int32)
        rewards    = np.array(batch.reward,   dtype=np.float32)
        terminals  = np.array(batch.terminal, dtype=np.float32)

        # Target Q-values (no gradient)
        q_next = self.target_network(next_states, training=False).numpy()  # (B, actions)
        targets = rewards + self.discount * np.max(q_next, axis=1) * (1.0 - terminals)
        targets = targets.astype(np.float32)

        with tf.GradientTape() as tape:
            q_values = self.main_network(states, training=True)             # (B, actions)
            action_masks = tf.one_hot(actions, self.num_actions)
            q_selected = tf.reduce_sum(q_values * action_masks, axis=1)    # (B,)
            loss = self.loss_fn(targets, q_selected)

        grads = tape.gradient(loss, self.main_network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.main_network.trainable_variables)
        )
        self.iterations += 1

        # Materialise log data *after* tape, using .numpy() to detach from graph
        loss_val = float(loss.numpy())
        mean_q   = float(tf.reduce_mean(q_values).numpy())

        for r in rewards:
            self.logger.log_step(float(r), loss_val, mean_q)

        if self.iterations % self.target_update_frequency == 0:
            self.target_network.set_weights(self.main_network.get_weights())
            print(f"[iter {self.iterations}] Target network updated.")

        if self.iterations % self.save_frequency == 0:
            self.checkpoint.save(file_prefix=self.checkpoint_path)
            self.logger.log_episode()
            self.logger.record(
                episode=self.iterations // self.save_frequency,
                epsilon=self.current_epsilon,
                step=self.iterations,
            )
            print(f"[iter {self.iterations}] Checkpoint saved.")


# Quick sanity-check
if __name__ == "__main__":
    net = create_network(NUM_ACTIONS)
    print(net.summary())

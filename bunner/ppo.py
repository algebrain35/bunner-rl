"""
PPO (Proximal Policy Optimization) agent for Infinite Bunner.

Drop-in replacement for dqn.py.  The CNN → LSTM backbone is identical;
the head is split into actor (policy) and critic (value) branches.

Usage in bunner.py:
    from ppo import PPO
    ppo = PPO(num_actions=5, ...)
    # then call  ppo.train_step(None, game.game_step)  each frame
"""

import tensorflow as tf
from tensorflow.keras import layers, models, Model
import numpy as np
import random
import os
from logger import MetricLogger


NUM_ACTIONS = 5  # 0=up, 1=right, 2=down, 3=left, 4=no-op


# ---------------------------------------------------------------------------
# State buffer  (identical to the DQN version)
# ---------------------------------------------------------------------------

class StateBuffer:
    """Rolling window of the last `seq_len` frames."""

    def __init__(self, seq_len: int = 4):
        self.seq_len = seq_len
        self.buffer: list[np.ndarray] = []

    def push(self, frame: np.ndarray):
        self.buffer.append(frame)
        if len(self.buffer) > self.seq_len:
            self.buffer.pop(0)

    def get_sequence(self) -> np.ndarray:
        """Returns (seq_len, 80, 80, 1).  Pads with oldest frame if needed."""
        if len(self.buffer) == 0:
            return np.zeros((self.seq_len, 80, 80, 1), dtype=np.float32)
        frames = list(self.buffer)
        if len(frames) < self.seq_len:
            pad = [frames[0]] * (self.seq_len - len(frames))
            frames = pad + frames
        return np.array(frames, dtype=np.float32)

    def is_ready(self) -> bool:
        return len(self.buffer) >= self.seq_len

    def reset(self):
        self.buffer.clear()

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# Network — shared CNN+LSTM backbone, separate actor & critic heads
# ---------------------------------------------------------------------------

def create_actor_critic(num_actions: int) -> Model:
    """
    DRQN-style backbone  →  policy logits  +  state value.
    Input: (batch, seq_len, 80, 80, 1)
    Outputs: logits (batch, num_actions), value (batch, 1)
    """
    inp = layers.Input(shape=(None, 80, 80, 1))

    # --- shared CNN + LSTM trunk (same architecture as the DQN) ---
    x = layers.TimeDistributed(
        layers.Conv2D(32, (8, 8), strides=4, activation="relu")
    )(inp)
    x = layers.TimeDistributed(
        layers.Conv2D(64, (4, 4), strides=2, activation="relu")
    )(x)
    x = layers.TimeDistributed(
        layers.Conv2D(64, (3, 3), strides=1, activation="relu")
    )(x)
    x = layers.TimeDistributed(layers.Flatten())(x)
    x = layers.LSTM(256, return_sequences=False)(x)
    x = layers.LayerNormalization()(x)

    # --- actor head (policy) ---
    actor = layers.Dense(256, activation="relu")(x)
    logits = layers.Dense(num_actions, name="logits")(actor)

    # --- critic head (value) ---
    critic = layers.Dense(256, activation="relu")(x)
    value = layers.Dense(1, name="value")(critic)

    return Model(inputs=inp, outputs=[logits, value])


# ---------------------------------------------------------------------------
# Rollout buffer — stores one horizon of on-policy experience
# ---------------------------------------------------------------------------

class RolloutBuffer:
    """Fixed-length buffer for a single PPO rollout."""

    def __init__(self):
        self.states: list[np.ndarray] = []
        self.actions: list[int] = []
        self.log_probs: list[float] = []
        self.rewards: list[float] = []
        self.values: list[float] = []
        self.terminals: list[bool] = []

    def store(self, state, action, log_prob, reward, value, terminal):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.terminals.append(terminal)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.rewards)


# ---------------------------------------------------------------------------
# PPO agent
# ---------------------------------------------------------------------------

class PPO:
    def __init__(
        self,
        num_actions: int = NUM_ACTIONS,
        learning_rate: float = 3e-4,
        discount_factor: float = 0.93,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        entropy_coeff: float = 0.01,
        value_coeff: float = 0.5,
        max_grad_norm: float = 0.5,
        horizon: int = 128,
        ppo_epochs: int = 4,
        mini_batch_size: int = 32,
        observation_limit: int = 1_000,
        save_frequency: int = 5_000,
        record_frequency: int = 10,
        seq_len: int = 4,
        checkpoint_dir: str = "./bunner/training_ppo",
        log_dir: str = "./bunner/logs_ppo",
    ):
        self.num_actions = num_actions
        self.discount = discount_factor
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_epsilon
        self.entropy_coeff = entropy_coeff
        self.value_coeff = value_coeff
        self.max_grad_norm = max_grad_norm
        self.horizon = horizon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.observation_limit = observation_limit
        self.save_frequency = save_frequency
        self.record_frequency = record_frequency
        self.checkpoint_dir = checkpoint_dir

        self.observations = 0
        self.iterations = 0
        self.episodes = 0

        self.state_buffer = StateBuffer(seq_len=seq_len)
        self.rollout = RolloutBuffer()

        self.optimizer = tf.keras.optimizers.Adam(
            learning_rate=learning_rate, clipnorm=max_grad_norm
        )
        self.network = create_actor_critic(num_actions)

        self.checkpoint = tf.train.Checkpoint(
            optimizer=self.optimizer, model=self.network
        )
        self.checkpoint_path = os.path.join(checkpoint_dir, "ppo_train")
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
            print(f"[PPO] Restored checkpoint: {latest}")
        else:
            print("[PPO] No checkpoint found — starting fresh.")

    def load_weights(self, filepath: str):
        self.network.load_weights(filepath)
        print(f"[PPO] Loaded weights from {filepath}")

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def choose_action(self, state_seq: np.ndarray):
        """
        Sample an action from the policy.
        state_seq: (1, seq_len, 80, 80, 1)
        Returns: action (int), log_prob (float), value (float)
        """
        logits, value = self.network(state_seq, training=False)
        dist = tf.random.categorical(logits, num_samples=1)  # (1, 1)
        action = int(dist[0, 0].numpy())

        log_probs = tf.nn.log_softmax(logits)
        log_prob = float(log_probs[0, action].numpy())
        val = float(value[0, 0].numpy())

        return action, log_prob, val

    # ------------------------------------------------------------------
    # GAE computation
    # ------------------------------------------------------------------

    def _compute_gae(self, last_value: float):
        """
        Compute Generalised Advantage Estimation over the current rollout.
        Returns numpy arrays of advantages and returns (same length as rollout).
        """
        buf = self.rollout
        n = len(buf)
        advantages = np.zeros(n, dtype=np.float32)
        returns = np.zeros(n, dtype=np.float32)

        gae = 0.0
        next_value = last_value
        for t in reversed(range(n)):
            mask = 0.0 if buf.terminals[t] else 1.0
            delta = (
                buf.rewards[t]
                + self.discount * next_value * mask
                - buf.values[t]
            )
            gae = delta + self.discount * self.gae_lambda * mask * gae
            advantages[t] = gae
            returns[t] = gae + buf.values[t]
            next_value = buf.values[t]

        return advantages, returns

    # ------------------------------------------------------------------
    # PPO optimisation  (multiple epochs over mini-batches)
    # ------------------------------------------------------------------

    def _optimize(self, last_value: float):
        advantages, returns = self._compute_gae(last_value)

        # Normalise advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std() + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        buf = self.rollout
        states = np.stack(buf.states).astype(np.float32)       # (N, seq, 80, 80, 1)
        actions = np.array(buf.actions, dtype=np.int32)         # (N,)
        old_log_probs = np.array(buf.log_probs, dtype=np.float32)

        n = len(buf)
        indices = np.arange(n)

        total_loss_accum = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            for start in range(0, n, self.mini_batch_size):
                end = start + self.mini_batch_size
                mb_idx = indices[start:end]
                if len(mb_idx) < 2:
                    continue

                mb_states = tf.constant(states[mb_idx])
                mb_actions = tf.constant(actions[mb_idx])
                mb_old_lp = tf.constant(old_log_probs[mb_idx])
                mb_adv = tf.constant(advantages[mb_idx])
                mb_ret = tf.constant(returns[mb_idx])

                loss_val = self._update_step(
                    mb_states, mb_actions, mb_old_lp, mb_adv, mb_ret
                )
                total_loss_accum += float(loss_val)
                num_updates += 1

        self.iterations += 1
        avg_loss = total_loss_accum / max(1, num_updates)
        mean_val = float(np.mean(buf.values))

        # Patch the most recent log_step entries with real loss/Q data
        self.logger.curr_ep_loss += avg_loss
        self.logger.curr_ep_q += mean_val
        self.logger.curr_ep_loss_length += 1

        # Write summary to log file every record_frequency iterations
        if self.iterations % self.record_frequency == 0:
            self.logger.record(
                episode=self.episodes,
                epsilon=0.0,
                step=self.iterations,
            )

        if self.iterations % self.save_frequency == 0:
            self.checkpoint.save(file_prefix=self.checkpoint_path)
            print(f"[PPO iter {self.iterations}] Checkpoint saved.")

    @tf.function
    def _update_step(self, states, actions, old_log_probs, advantages, returns):
        with tf.GradientTape() as tape:
            logits, values = self.network(states, training=True)
            values = tf.squeeze(values, axis=-1)  # (B,)

            # --- Policy loss (clipped surrogate) ---
            log_probs = tf.nn.log_softmax(logits)
            action_masks = tf.one_hot(actions, self.num_actions)
            new_log_probs = tf.reduce_sum(log_probs * action_masks, axis=1)

            ratio = tf.exp(new_log_probs - old_log_probs)
            clipped_ratio = tf.clip_by_value(
                ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps
            )
            policy_loss = -tf.reduce_mean(
                tf.minimum(ratio * advantages, clipped_ratio * advantages)
            )

            # --- Value loss ---
            value_loss = tf.reduce_mean(tf.square(returns - values))

            # --- Entropy bonus ---
            probs = tf.nn.softmax(logits)
            entropy = -tf.reduce_sum(probs * log_probs, axis=1)
            entropy_bonus = tf.reduce_mean(entropy)

            # --- Total loss ---
            loss = (
                policy_loss
                + self.value_coeff * value_loss
                - self.entropy_coeff * entropy_bonus
            )

        grads = tape.gradient(loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.network.trainable_variables)
        )
        return loss

    # ------------------------------------------------------------------
    # Main entry point — called once per game action
    # ------------------------------------------------------------------

    def train_step(self, _state, game_step) -> np.ndarray:
        """
        1. Build current state sequence from buffer.
        2. Choose action via policy.
        3. Step the game.
        4. Store transition in rollout buffer.
        5. If horizon reached, run PPO update.
        Returns the next frame (80, 80, 1).
        """
        current_seq = self.state_buffer.get_sequence()       # (seq_len, 80, 80, 1)
        batch_seq = np.expand_dims(current_seq, axis=0)      # (1, seq_len, 80, 80, 1)

        action, log_prob, value = self.choose_action(batch_seq)

        self.observations += 1
        print(
            f"[PPO] obs={self.observations} | "
            f"action={action} | "
            f"rollout={len(self.rollout)}/{self.horizon}"
        )

        # game_step pushes the new frame into state_buffer internally
        next_frame, reward, terminal = game_step(action)

        # Log every step's reward (loss=0 when not optimising)
        self.logger.log_step(reward, 0, 0)

        self.rollout.store(current_seq, action, log_prob, reward, value, terminal)

        # When we've collected enough steps, optimise
        if len(self.rollout) >= self.horizon or terminal:
            if self.observations > self.observation_limit:
                # Bootstrap value for the last state (0 if terminal)
                if terminal:
                    last_value = 0.0
                else:
                    next_seq = self.state_buffer.get_sequence()
                    _, lv = self.network(
                        np.expand_dims(next_seq, 0), training=False
                    )
                    last_value = float(lv[0, 0].numpy())
                self._optimize(last_value)
            self.rollout.clear()

        # Mark end of episode on death so the logger flushes accumulators
        if terminal:
            self.episodes += 1
            self.logger.log_episode()

        return next_frame


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    net = create_actor_critic(NUM_ACTIONS)
    net.summary()

    # Smoke test forward pass
    dummy = np.random.rand(2, 4, 80, 80, 1).astype(np.float32)
    logits, values = net(dummy)
    print(f"logits shape: {logits.shape}")   # (2, 5)
    print(f"values shape: {values.shape}")   # (2, 1)

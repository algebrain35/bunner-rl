from types import NoneType, new_class
from typing import final
import tensorflow as tf
from tensorflow.keras import layers, models
import random
from collections import namedtuple, deque
from tqdm import tqdm
import numpy as np
import os
from logger import MetricLogger

NUM_ACTIONS = 4

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminal'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    def __len__(self):
        return len(self.memory)

class DQN:
    def __init__(self,
                 num_actions,
                 replay_size,
                 learning_rate,
                 discount_factor,
                 init_epsilon,
                 final_epsilon,
                 epsilon_decay,
                 observation_limit):
        self.replay = ReplayMemory(replay_size)
        self.discount = discount_factor
        self.eps = init_epsilon
        self.eps_final = final_epsilon
        self.eps_decay = epsilon_decay
        self.lr = learning_rate
        self.num_actions = num_actions
        self.loss_fn = tf.keras.losses.Huber()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.main_network = create_network(num_actions)
        self.target_network = create_network(num_actions)
        self.observations = 0
        self.observation_limit = observation_limit
        self.batch_size = 256
        self.target_update_frequency = 1000
        self.iterations = 0
        self.save_frequency = 1000
        self.checkpoint = tf.train.Checkpoint(optimizer=self.optimizer, model=self.main_network)
        self.checkpoint_dir = "./bunner/training"
        self.restore_checkpoint()
        self.checkpoint_path = os.path.join(self.checkpoint_dir, "dqn_train")
        self.logger = MetricLogger("./bunner/logs")
        self.threshold = self.eps
    def restore_checkpoint(self):
        assert os.path.exists(self.checkpoint_dir)
        self.checkpoint.restore(tf.train.latest_checkpoint(self.checkpoint_dir))    
    def load_weights(self, filepath):
        self.target_network.load_weights(filepath)
        self.target_network.load_weights(filepath)
    def preprocess(self, X):
        if np.shape(X) == (80, 80, 4):
            return np.expand_dims(X, axis=0)

    def choose_action(self, state):
        rand_num = random.random()
        threshold = self.eps_final + (self.eps - self.eps_final) * tf.exp(-self.iterations / self.eps_decay)
        threshold = max(self.eps_final, threshold)
        self.threshold = threshold
        

        self.observations += 1
        if rand_num > threshold:
            Q_values = self.main_network.predict(self.preprocess(state))
            return tf.argmax(Q_values[0]).numpy()
        else:
            return random.randrange(self.num_actions)

    def train_step(self, state, game_step):
        threshold = self.eps_final + (self.eps - self.eps_final) * tf.exp(-1 * self.iterations / self.eps_decay)
        
        
        action = self.choose_action(state)
        print(f"Epsilon: {threshold} Observations: {self.observations} Action: {action}")
        next_state, reward, terminal = game_step(action)
        self.replay.push(
                state,
                action,
                next_state,
                reward,
                terminal)


        if self.observations > self.observation_limit and len(self.replay) >= self.batch_size:
            samples = self.replay.sample(self.batch_size)
            batch = Transition(*zip(*samples))
            #print(batch)

            '''
            for i, s in enumerate(batch.state):
                if not np.any(s): 
                    print(f"Index {i} is BROKEN! Shape is {s}")
                elif s.shape != (80, 80, 4):
                    print(f"Index {i} is BROKEN! Shape is {s.shape}")
            '''


            actions = np.array(batch.action, dtype=np.int32)
            rewards = np.array(batch.reward, dtype=np.float32)
            terminals = np.array(batch.terminal, dtype=np.float32)
            states = np.stack(batch.state)
            new_states = np.stack(batch.next_state)

            Q_next = self.target_network.predict(new_states)
            targets = rewards + self.discount * np.max(Q_next, axis=1) * (1 - terminals)
            
            log_data = None
            with tf.GradientTape() as tape:
                Q_values = self.main_network(states)
                action_masks = tf.one_hot(actions, self.num_actions)
                Q_selected = tf.reduce_sum(Q_values * action_masks, axis=1)
                loss = self.loss_fn(targets, Q_selected)
                log_data = zip(rewards, Q_values)
            grads = tape.gradient(loss, self.main_network.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.main_network.trainable_variables))
            self.iterations += 1
            
            for reward, Q in log_data:
                self.logger.log_step(reward, loss, Q)
            if self.iterations % self.target_update_frequency == 0:
                self.target_network.set_weights(self.main_network.get_weights())
            if self.iterations % self.save_frequency == 0:
                self.checkpoint.save(file_prefix=self.checkpoint_path)
                self.logger.log_episode()
                self.logger.record(episode=self.iterations//self.save_frequency, epsilon=self.threshold, step=self.iterations)

        return next_state

def create_network(num_actions):
    model = models.Sequential([
        layers.Input(shape=(80, 80, 4)),
        layers.Conv2D(32, (8, 8), strides=4, padding="same", activation="relu"),
        layers.MaxPooling2D(pool_size=(2, 2), padding="same"),
        layers.Conv2D(64, (4, 4), strides=2, padding="same", activation="relu"),
        layers.Conv2D(64, (3, 3), strides=1, padding="same", activation="relu"),
        layers.Flatten(),
        layers.Dense(512, activation="relu", name="hidden_fc1"),
        layers.Dense(num_actions, activation="linear", name="readout")
    ])

    return model

model = create_network(NUM_ACTIONS)
print(model.summary())
rng = np.random.default_rng()
test = rng.random(size=(80, 80, 4))
test = np.expand_dims(test, axis=0)

print(model(test))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import time
import os
import json
from collections import deque
from tqdm import trange


class DQNTrainer:
    class DQN(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 128)
            self.fc2 = nn.Linear(128, 128)
            self.out = nn.Linear(128, output_dim)

        def forward(self, x):
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.out(x)

    class ReplayBuffer:
        def __init__(self, capacity=50000):
            self.buffer = deque(maxlen=capacity)

        def push(self, state, action, reward, next_state, done):
            self.buffer.append((state, action, float(reward), next_state, done))

        def sample(self, batch_size):
            samples = random.sample(self.buffer, batch_size)
            states, actions, rewards, next_states, dones = zip(*samples)

            return (
                torch.tensor(np.stack(states), dtype=torch.float32),
                torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
                torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
                torch.tensor(np.stack(next_states), dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
            )

        def __len__(self):
            return len(self.buffer)

    def __init__(self, env, model_path=None, buffer_capacity=50000):
        self.env = env
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[INFO] Using device: {self.device}")
        self.model_path = model_path
        self.pretrained_loaded = False
        self.trained_episodes = None

        input_dim = len(env.state_space)
        output_dim = len(env.action_space)

        self.policy_net = self.DQN(input_dim, output_dim).to(self.device)
        self.target_net = self.DQN(input_dim, output_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=1e-4)
        self.buffer = self.ReplayBuffer(capacity=buffer_capacity)

        if model_path and os.path.exists(model_path):
            self._load_model()

    def _load_model(self):
        self.policy_net.load_state_dict(torch.load(self.model_path, map_location=self.device))
        self.policy_net.eval()
        self.pretrained_loaded = True
        print(f"[INFO] Loaded pretrained model from {self.model_path}")

        meta_path = self.model_path.replace(".pth", "_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r") as f:
                self.trained_episodes = json.load(f).get("episodes")
                print(f"[INFO] Metadata: Trained Episodes = {self.trained_episodes}")

    def train(self, episodes=300, gamma=0.99, epsilon_start=1.0, epsilon_end=0.05,
              epsilon_decay=0.99, batch_size=128, target_update_freq=5):

        if self.trained_episodes == episodes and self.pretrained_loaded:
            print(f"[INFO] Skipping training. Model already trained for {episodes} episodes.")
            return self.policy_net, [], [], {}

        print(f"[INFO] Training model from scratch for {episodes} episodes...")

        epsilon = epsilon_start
        max_reward = -float('inf')
        reward_history = []
        best_state_action_pairs = []
        action_counter = {a: 0 for a in self.env.action_space}

        start_time = time.time()

        for episode in trange(episodes, desc="Training Episodes"):
            state = self.env.reset()
            total_reward = 0
            done = False

            while not done:
                if np.random.rand() < epsilon:
                    action = np.random.choice(self.env.action_space)
                else:
                    with torch.no_grad():
                        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                        q_values = self.policy_net(state_tensor)
                        action = torch.argmax(q_values).item()

                next_state, reward, done, _ = self.env.step(action)
                self.buffer.push(state, action, reward, next_state, done)
                total_reward += reward
                action_counter[action] += 1
                state = next_state

                if len(self.buffer) >= batch_size:
                    self._optimize_model(batch_size, gamma)

            if episode % target_update_freq == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            if total_reward > max_reward:
                max_reward = total_reward
                best_state_action_pairs.append((state.copy(), action))
                if self.model_path:
                    torch.save(self.policy_net.state_dict(), self.model_path)

            reward_history.append(total_reward)
            epsilon = max(epsilon_end, epsilon * epsilon_decay)

            if os.environ.get("DEBUG", "0") == "1":
                print(f"[DEBUG] Episode {episode+1} | Reward: {total_reward:.3f} | Buffer: {len(self.buffer)} | Action Count: {action_counter}")

        duration = time.time() - start_time
        print(f"\nTraining complete in {duration / 60:.2f} minutes")
        print("Best Episode Reward :", max_reward)
        print("Final Action Count   :", action_counter)

        if self.model_path:
            meta_path = self.model_path.replace(".pth", "_meta.json")
            with open(meta_path, "w") as f:
                json.dump({"episodes": episodes}, f)

        return self.policy_net, best_state_action_pairs, reward_history, action_counter

    def _optimize_model(self, batch_size, gamma):
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_q = self.target_net(next_states).gather(1, self.policy_net(next_states).argmax(1, keepdim=True))
            target_q = rewards + gamma * next_q * (1 - dones)

        current_q = self.policy_net(states).gather(1, actions)
        loss = F.mse_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()
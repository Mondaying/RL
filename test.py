import gymnasium as gym
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim


# ======================
# 1. Q Network
# ======================
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ======================
# 2. Replay Buffer
# ======================
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, s_, d):
        self.buffer.append((s, a, r, s_, d))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, s_, d = map(np.array, zip(*batch))
        return s, a, r, s_, d

    def __len__(self):
        return len(self.buffer)


# ======================
# 3. Environment
# ======================
env = gym.make("CartPole-v1")

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

q_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(q_net.state_dict())

optimizer = optim.Adam(q_net.parameters(), lr=1e-3)
buffer = ReplayBuffer()


# ======================
# 4. Hyperparameters
# ======================
gamma = 0.99
batch_size = 64

epsilon = 1.0
epsilon_min = 0.1
epsilon_decay = 0.995

tau = 0.005   # soft update

episodes = 1000


# ======================
# state normalization（关键！）
# ======================
state_scale = np.array([2.4, 2.0, 0.21, 2.0])


# ======================
# 5. Training
# ======================
rewards_history = []
global_step = 0

for ep in range(episodes):

    state, _ = env.reset()
    state = state / state_scale   # ⭐ normalize

    total_reward = 0

    while True:
        global_step += 1

        # ε-greedy
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            s_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = q_net(s_tensor).argmax(dim=1).item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        next_state = next_state / state_scale  # ⭐ normalize

        buffer.push(state, action, reward, next_state, done)

        state = next_state
        total_reward += reward

        # ======================
        # training
        # ======================
        if len(buffer) > batch_size:

            s, a, r, s_, d = buffer.sample(batch_size)

            s = torch.FloatTensor(s)
            a = torch.LongTensor(a)
            r = torch.FloatTensor(r)
            s_ = torch.FloatTensor(s_)
            d = torch.FloatTensor(d)

            q_values = q_net(s).gather(1, a.unsqueeze(1)).squeeze()

            # ⭐ Double DQN (核心稳定点)
            with torch.no_grad():
                next_actions = q_net(s_).argmax(1, keepdim=True)
                next_q = target_net(s_).gather(1, next_actions).squeeze()

                target = r + gamma * next_q * (1 - d)

            loss = nn.MSELoss()(q_values, target)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(q_net.parameters(), 10)
            optimizer.step()

        # ======================
        # soft update target net
        # ======================
        for tp, p in zip(target_net.parameters(), q_net.parameters()):
            tp.data.copy_(tau * p.data + (1 - tau) * tp.data)

        if done:
            break

    # ======================
    # epsilon schedule（防塌缩关键）
    # ======================
    epsilon = max(epsilon_min, epsilon * epsilon_decay)

    rewards_history.append(total_reward)

    print(f"Episode {ep}, Reward: {total_reward}, Epsilon: {epsilon:.3f}")


# ======================
# 6. plot
# ======================
plt.plot(rewards_history)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Stable DQN Training Curve")
plt.show()


# ======================
# 7. test render
# ======================
env = gym.make("CartPole-v1", render_mode="rgb_array")
state, _ = env.reset()
state = state / state_scale

while True:
    s_tensor = torch.FloatTensor(state).unsqueeze(0)
    action = q_net(s_tensor).argmax(dim=1).item()

    state, reward, terminated, truncated, _ = env.step(action)
    state = state / state_scale

    frame = env.render()
    plt.imshow(frame)
    plt.axis("off")
    plt.pause(0.01)
    plt.clf()

    if terminated or truncated:
        break

env.close()
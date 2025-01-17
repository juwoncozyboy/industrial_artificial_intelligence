
import random
import math
from collections import deque
from itertools import accumulate
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Original parameters and data from the assignment
PROCESSING_TIMES = {
    'A': [5, 5],
    'B': [3, 3, 3],
    'C': [7, 7],
    'D': [4],
    'E': [2],
    'F': [6, 6, 6]
}

DUE_DATES = {
    'A': [10, 10],
    'B': [15, 15, 15],
    'C': [25, 25],
    'D': [12],
    'E': [14],
    'F': [21, 21, 21]
}

# 학습 파라미터
N_EPISODES = 75000
ALPHA = 0.1
GAMMA = 0.95
EPSILON = 1.0
EPSILON_DECAY = 0.9995
EPSILON_MIN = 0.01

# Neural Network Model (from DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer Class (from DQN)
class ReplayBuffer:
    def __init__(self, buffer_limit=5000):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, transition):
        self.buffer.append(transition)

    def sample(self, n):
        mini_batch = random.sample(self.buffer, n)
        s_lst, a_lst, r_lst, s_prime_lst, done_mask_lst = [], [], [], [], []

        for transition in mini_batch:
            s, a, r, s_prime, done_mask = transition
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            done_mask_lst.append([done_mask])

        return torch.tensor(s_lst, dtype=torch.float),                torch.tensor(a_lst),                torch.tensor(r_lst, dtype=torch.float),                torch.tensor(s_prime_lst, dtype=torch.float),                torch.tensor(done_mask_lst, dtype=torch.float)

    def size(self):
        return len(self.buffer)

# DQN Agent Implementation
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN
        self.gamma = GAMMA
        self.learning_rate = 0.001
        self.batch_size = 64

        self.model = DQN(state_size, action_size)
        self.target_model = DQN(state_size, action_size)
        self.update_target_model()

        self.memory = ReplayBuffer()

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, next_state, done):
        self.memory.put((state, action, reward, next_state, done))

    def act(self, state):
        if random.random() <= self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            q_values = self.model(state)
            return torch.argmax(q_values).item()

    def replay(self):
        if self.memory.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        curr_q_values = self.model(states).gather(1, actions)
        next_q_values = self.target_model(next_states).max(1)[0].unsqueeze(1)
        target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        loss = F.mse_loss(curr_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def train(self, environment):
        for episode in range(N_EPISODES):
            state = environment.reset()
            total_reward = 0

            done = False
            while not done:
                action = self.act(state)
                next_state, reward, done, _ = environment.step(action)

                total_reward += reward
                self.remember(state, action, reward, next_state, done)

                state = next_state
                self.replay()

            if self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

            self.update_target_model()

            if episode % 100 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {self.epsilon}")

# Assuming environment is similar to original SARSA/Q-learning

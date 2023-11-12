from datetime import datetime
import torch

from os import path
import numpy as np
import random
from collections import deque


class QNetwork(torch.nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.fc = torch.nn.Linear(state_size, 48)
        self.fcQ1 = torch.nn.Linear(48, 64)
        self.fcQ2 = torch.nn.Linear(64, action_size)

    def forward(self, x):
        x = self.fc(x)
        x = torch.nn.functional.relu(x)
        x = self.fcQ1(x)
        x = torch.nn.functional.relu(x)
        x = self.fcQ2(x)

        return x


class Agent:
    def __init__(self, state_size, is_eval=False, model_name=""):
        self.state_size = state_size  # normalized previous days
        self.action_size = 3  # sit, buy, sell
        self.memory = deque(maxlen=10000)
        self.inventory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update_period = 4

        self.target_network = QNetwork(self.state_size, self.action_size)
        self.online_network = QNetwork(self.state_size, self.action_size)

        if is_eval:
            self.load()
            self.online_network.eval()

        self.optimizer = torch.optim.Adam(self.online_network.parameters(), lr=0.0005)

    def save(self, e):
        torch.save(self.target_network.state_dict(), "models/model_ep" + str(e) + ".pt")

    def load(self):
        self.online_network.load_state_dict(torch.load("models/" + self.model_name + ".pt"))

    def act(self, state):

        if not self.is_eval and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)

        with torch.no_grad():
            state = torch.tensor(state, dtype=torch.float)
            options = self.online_network(state)

        return torch.argmax(options[0]).item()  # options.shape : (window_size, actoin_size)

    def update_target(self):
        self.target_network.load_state_dict(self.online_network.state_dict())

    def expReplay(self, batch_size):

        loss = 0

        for state, action, reward, state_next, done in random.choices(self.memory, k=batch_size):
            state = torch.tensor(state, dtype=torch.float)
            action = torch.tensor(action, dtype=torch.long)
            state_next = torch.tensor(state_next, dtype=torch.float)
            reward = torch.tensor(reward, dtype=torch.float)
            done = torch.tensor(done, dtype=torch.float)

            # print(f"state : {state}")
            # print(f"action : {action}")
            # print(f"state_next : {state_next}")

            with torch.no_grad():
                target = reward + (1 - done) * self.gamma * torch.max(self.target_network(state_next))

            loss += (target - self.online_network(state)[0][action]) ** 2

        loss /= batch_size

        # Initialize gradient
        self.optimizer.zero_grad()

        # Calculate gradient
        loss.backward()

        # Take a gradient descent step
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
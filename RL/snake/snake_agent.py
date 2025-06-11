import torch
import random
import numpy as np
from collections import deque
from snake_game import SnakeGameAI, Direction
import matplotlib.pyplot as plt
from IPython import display

class QLearnAgent:
    def __init__(self, state_size=11, action_size=3, hidden_size=256, lr=0.001):
        self.state_size = state_size  # 11 binary values
        self.action_size = action_size  # [straight, right, left]
        self.memory = deque(maxlen=100000)
        self.gamma = 0.9  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = lr
        self.model = self._build_model(hidden_size)
        self.trainer = self._get_trainer()
    
    def _build_model(self, hidden_size):
        # Create a simple neural network
        model = torch.nn.Sequential(
            torch.nn.Linear(self.state_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, self.action_size)
        )
        return model
    
    def _get_trainer(self):
        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
    
    def remember(self, state, action, reward, next_state, done):
        # Store experience in memory
        self.memory.append((state, action, reward, next_state, done))
    
    def get_action(self, state):
        # Epsilon-greedy action selection
        if random.random() < self.epsilon:
            # Exploration: choose random action
            action = random.randint(0, self.action_size - 1)
            final_move = [0] * self.action_size
            final_move[action] = 1
            return final_move
        
        # Exploitation: choose best action based on Q-values
        state0 = torch.tensor(state, dtype=torch.float)
        prediction = self.model(state0)
        action = torch.argmax(prediction).item()
        final_move = [0] * self.action_size
        final_move[action] = 1
        return final_move
    
    def train_step(self, state, action, reward, next_state, done):
        # Convert data to tensors
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        
        # Get current Q values
        current_q = self.model(state)
        
        # Q-learning formula: Q(s,a) = r + gamma * max(Q(s',a'))
        target = current_q.clone()
        if done:
            target[torch.argmax(action).item()] = reward
        else:
            next_q = self.model(next_state)
            target[torch.argmax(action).item()] = reward + self.gamma * torch.max(next_q)
        
        # Train the network
        self.trainer.zero_grad()
        loss = torch.nn.functional.mse_loss(current_q, target)
        loss.backward()
        self.trainer.step()
        
        # Decay epsilon for less exploration over time
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def train_from_memory(self, batch_size=64):
        # Train on random samples from memory (experience replay)
        if len(self.memory) > batch_size:
            mini_batch = random.sample(self.memory, batch_size)
            for state, action, reward, next_state, done in mini_batch:
                self.train_step(state, action, reward, next_state, done)

    def save(self, file_name='model.pth'):
        torch.save(self.model.state_dict(), file_name)
    
    def load(self, file_name='model.pth'):
        self.model.load_state_dict(torch.load(file_name))
        self.model.eval()
import torch
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from IPython import display
from snake_game import SnakeGameAI
from snake_agent import QLearnAgent

def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.1)

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = QLearnAgent()
    game = SnakeGameAI(width=300, height=300, block_size=50)
    
    while True:
        # Get current state
        state_old = game.reset()
        
        done = False
        score = 0
        
        while not done:
            # Get action from agent
            action = agent.get_action(state_old)
            
            # Perform action and get new state
            reward, done, score = game.play_step(action)
            state_new = game._get_state()
            
            # Train the agent
            agent.train_step(state_old, action, reward, state_new, done)
            agent.remember(state_old, action, reward, state_new, done)
            
            state_old = state_new
            
            # Train on past experiences (experience replay)
            agent.train_from_memory()
            
            if done:
                # Game over, training episode completed
                game.reset()
                
                # Update scores for plotting
                plot_scores.append(score)
                total_score += score
                mean_score = total_score / len(plot_scores)
                plot_mean_scores.append(mean_score)
                
                # Update record
                if score > record:
                    record = score
                    agent.save()
                
                # Plot progress
                if len(plot_scores) % 10 == 0:
                    print(f'Game: {len(plot_scores)}, Score: {score}, Average: {mean_score:.1f}, Record: {record}')
                    plot(plot_scores, plot_mean_scores)
                break

if __name__ == '__main__':
    train()
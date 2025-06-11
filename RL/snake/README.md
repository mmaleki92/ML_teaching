# Snake Game with Reinforcement Learning

This project implements a simple snake game using Pygame and trains a Q-learning agent to play it.

## Requirements
- Python 3.6+
- Pygame
- NumPy

## Installation
```bash
pip install pygame numpy
```

## Usage
1. Run the training script:
```bash
python train.py
```

2. Watch the pre-trained model:
```bash
python play.py
```

## How it works
The agent uses Q-learning to learn optimal actions in different states of the game.
- **State**: Relative position of food + danger information (obstacles in 4 directions)
- **Actions**: Move up, down, left, or right
- **Reward**: +10 for eating food, -10 for game over, -0.1 for each step (to encourage efficient paths)
```
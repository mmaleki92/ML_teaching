import pygame
import random
import numpy as np
from enum import Enum

class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

class SnakeGameAI:
    def __init__(self, width=640, height=480, block_size=20):
        # Pygame initialization
        pygame.init()
        self.width = width
        self.height = height
        self.block_size = block_size
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake Game AI')
        self.clock = pygame.time.Clock()
        
        # Game state
        self.reset()
    
    def reset(self):
        # Initialize game state
        self.direction = Direction.RIGHT
        
        self.head = [self.width // 2, self.height // 2]
        self.snake = [
            self.head,
            [self.head[0] - self.block_size, self.head[1]],
            [self.head[0] - 2 * self.block_size, self.head[1]]
        ]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        
        return self._get_state()
    
    def _place_food(self):
        x = random.randint(0, (self.width - self.block_size) // self.block_size) * self.block_size
        y = random.randint(0, (self.height - self.block_size) // self.block_size) * self.block_size
        self.food = [x, y]
        if self.food in self.snake:
            self._place_food()
    
    def play_step(self, action):
        self.frame_iteration += 1
        reward = 0
        game_over = False
        
        # 1. Handle events (for manual control)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        
        # 2. Move
        self._move(action)
        self.snake.insert(0, list(self.head))
        
        # 3. Check if game over
        if self._is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # 4. Place new food or move
        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()
            reward = -0.1  # Small penalty for each step
        
        # 5. Update UI and clock
        self._update_ui()
        self.clock.tick(10)  # Control game speed
        
        return reward, game_over, self.score
    
    def _is_collision(self):
        # Hit boundary
        if (self.head[0] >= self.width or self.head[0] < 0 or
                self.head[1] >= self.height or self.head[1] < 0):
            return True
        
        # Hit itself
        if self.head in self.snake[1:]:
            return True
        
        return False
    
    def _update_ui(self):
        self.display.fill((0, 0, 0))  # Black background
        
        # Draw snake
        for pt in self.snake:
            pygame.draw.rect(self.display, (0, 255, 0), pygame.Rect(pt[0], pt[1], self.block_size, self.block_size))
            pygame.draw.rect(self.display, (50, 205, 50), pygame.Rect(pt[0] + 4, pt[1] + 4, 12, 12))
        
        # Draw food
        pygame.draw.rect(self.display, (255, 0, 0), pygame.Rect(self.food[0], self.food[1], self.block_size, self.block_size))
        
        # Draw score
        font = pygame.font.SysFont('arial', 25)
        text = font.render(f"Score: {self.score}", True, (255, 255, 255))
        self.display.blit(text, [0, 0])
        pygame.display.flip()
    
    def _move(self, action):
        # [straight, right, left]
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        
        if np.array_equal(action, [1, 0, 0]):
            # Keep direction (straight)
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            # Turn right
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            # Turn left
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]
        
        self.direction = new_dir
        
        # Update head position
        x = self.head[0]
        y = self.head[1]
        if self.direction == Direction.RIGHT:
            x += self.block_size
        elif self.direction == Direction.LEFT:
            x -= self.block_size
        elif self.direction == Direction.DOWN:
            y += self.block_size
        elif self.direction == Direction.UP:
            y -= self.block_size
        
        self.head = [x, y]
    
    def _get_state(self):
        head = self.snake[0]
        
        # Get points around the head
        point_l = [head[0] - self.block_size, head[1]]
        point_r = [head[0] + self.block_size, head[1]]
        point_u = [head[0], head[1] - self.block_size]
        point_d = [head[0], head[1] + self.block_size]
        
        # Check if current direction
        dir_l = self.direction == Direction.LEFT
        dir_r = self.direction == Direction.RIGHT
        dir_u = self.direction == Direction.UP
        dir_d = self.direction == Direction.DOWN
        
        # Check for danger
        danger_straight = (dir_r and self._is_danger(point_r)) or \
                         (dir_l and self._is_danger(point_l)) or \
                         (dir_u and self._is_danger(point_u)) or \
                         (dir_d and self._is_danger(point_d))
                         
        danger_right = (dir_r and self._is_danger(point_d)) or \
                      (dir_l and self._is_danger(point_u)) or \
                      (dir_u and self._is_danger(point_r)) or \
                      (dir_d and self._is_danger(point_l))
                      
        danger_left = (dir_r and self._is_danger(point_u)) or \
                     (dir_l and self._is_danger(point_d)) or \
                     (dir_u and self._is_danger(point_l)) or \
                     (dir_d and self._is_danger(point_r))
        
        # Food direction
        food_left = self.food[0] < head[0]
        food_right = self.food[0] > head[0]
        food_up = self.food[1] < head[1]
        food_down = self.food[1] > head[1]
        
        # State as binary values
        state = [
            # Danger
            danger_straight,
            danger_right,
            danger_left,
            
            # Direction
            dir_l, dir_r, dir_u, dir_d,
            
            # Food location
            food_left, food_right, food_up, food_down
        ]
        
        return np.array(state, dtype=int)
    
    def _is_danger(self, point):
        # Check if point is out of bounds or snake body
        if (point[0] >= self.width or point[0] < 0 or
                point[1] >= self.height or point[1] < 0 or
                point in self.snake[1:]):
            return True
        return False
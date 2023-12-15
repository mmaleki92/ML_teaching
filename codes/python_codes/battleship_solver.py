import numpy as np
import math
import random
import matplotlib.pyplot as plt
import seaborn as sns

class BattleshipGame:
    def __init__(self, grid_size=10, ships=None):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=int)
        self.ships = ships if ships else [5, 4, 3, 3, 2]
        self.place_ships()

    def place_ships(self):
        for ship in self.ships:
            placed = False
            while not placed:
                row, col = random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1)
                horiz = random.choice([True, False])
                if self.can_place_ship(row, col, ship, horiz):
                    self.set_ship(row, col, ship, horiz)
                    placed = True

    def can_place_ship(self, row, col, size, horiz):
        if horiz:
            if col + size > self.grid_size:
                return False
            return np.all(self.grid[row, col:col + size] == 0)
        else:
            if row + size > self.grid_size:
                return False
            return np.all(self.grid[row:row + size, col] == 0)

    def set_ship(self, row, col, size, horiz):
        if horiz:
            self.grid[row, col:col + size] = 1
        else:
            self.grid[row:row + size, col] = 1

    def check_hit(self, row, col):
        return self.grid[row, col] == 1

    def print_grid(self):
        print(self.grid)

class BattleshipSolver:
    def __init__(self, grid_size=10, ships=None):
        self.grid_size = grid_size
        self.grid = np.full((grid_size, grid_size), -1)  # -1 for unknown, 0 for miss, 1 for hit
        self.ships = ships if ships else [5, 4, 3, 3, 2]  # Default ship sizes
        self.ship_hits = {size: 0 for size in self.ships}  # Track hits on each ship

    def plot_grid(self, battleship_game, information_gain_grid):
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # Plot for solver's guesses
        sns.heatmap(self.grid, annot=True, cmap="coolwarm", cbar=False, fmt="d", ax=axes[0])
        axes[0].set_title("Solver's Guesses (Hits/Misses)")
        axes[0].set_xlabel("Column")
        axes[0].set_ylabel("Row")

        # Plot for actual ship placements
        sns.heatmap(battleship_game.grid, annot=True, cmap="YlGn", cbar=False, ax=axes[1])
        axes[1].set_title("Actual Ship Placements")
        axes[1].set_xlabel("Column")
        axes[1].set_ylabel("Row")

        # Plot for information gain
        sns.heatmap(information_gain_grid, annot=True, cmap="viridis", cbar=True, ax=axes[2])
        axes[2].set_title("Expected Information Gain")
        axes[2].set_xlabel("Column")
        axes[2].set_ylabel("Row")

        plt.tight_layout()
        plt.show()

    def calculate_information_gain(self):
        current_entropy = self.calculate_entropy()
        information_gain_grid = np.zeros((self.grid_size, self.grid_size))

        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.grid[row, col] == -1:  # Only calculate for unknown cells
                    # Simulate a hit
                    self.grid[row, col] = 1
                    new_entropy = self.calculate_entropy()
                    self.grid[row, col] = -1  # Reset back to unknown

                    # Information gain is the reduction in entropy
                    information_gain = np.sum(current_entropy) - np.sum(new_entropy)
                    information_gain_grid[row, col] = information_gain

        return information_gain_grid
    
    def plot_information_gain(self, information_gain_grid):
        plt.figure(figsize=(10, 6))
        sns.heatmap(information_gain_grid, annot=True, cmap="viridis", cbar=True)
        plt.title("Expected Information Gain")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.show()

    def calculate_entropy(self):
        entropy_grid = np.zeros((self.grid_size, self.grid_size))
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.grid[row, col] == -1:  # Calculate only for unknown cells
                    prob_ship = self.calculate_ship_probability(row, col)
                    entropy = 0
                    if 0 < prob_ship < 1:
                        entropy = -prob_ship * math.log2(prob_ship) - \
                                (1 - prob_ship) * math.log2(1 - prob_ship)
                    entropy_grid[row, col] = entropy
        return entropy_grid


    def calculate_ship_probability(self, row, col):
        if self.grid[row, col] != -1:  # If already guessed, probability is 0
            return 0

        total_positions = 0
        for ship_size in self.ships:
            # Check horizontal and vertical positions for the ship
            for dr, dc in [(0, 1), (1, 0)]:
                for shift in range(ship_size):
                    r_start = row - dr * shift
                    c_start = col - dc * shift
                    r_end = r_start + dr * ship_size
                    c_end = c_start + dc * ship_size

                    if 0 <= r_start < self.grid_size and 0 <= c_start < self.grid_size and \
                       0 <= r_end <= self.grid_size and 0 <= c_end <= self.grid_size:
                        if self.is_valid_position(r_start, c_start, ship_size, dr == 0):
                            total_positions += 1

        # Normalize by the total number of positions
        return total_positions / self.total_possible_positions()
    
    def is_valid_position(self, row, col, ship_size, horizontal):
        # Check if ship placement is valid considering hits and misses
        for i in range(ship_size):
            r, c = (row, col + i) if horizontal else (row + i, col)
            if self.grid[r, c] == 0 or r >= self.grid_size or c >= self.grid_size:
                return False
        return True
    def total_possible_positions(self):
        # Calculate the total number of possible ship positions on the grid
        total = 0
        for ship_size in self.ships:
            total += (self.grid_size - ship_size + 1) * self.grid_size * 2
        return total

    def make_guess(self, entropy_grid):
        max_entropy = np.unravel_index(np.argmax(entropy_grid, axis=None), entropy_grid.shape)
        return max_entropy  # Returns the coordinates with the highest entropy

    def update_grid(self, row, col, result):
        self.grid[row, col] = 1 if result == 'hit' else 0
        if result == 'hit':
            for ship_size, hits in self.ship_hits.items():
                if hits < ship_size:  # Update the first ship that isn't sunk yet
                    self.ship_hits[ship_size] += 1
                    break

    def is_game_over(self):
        return all(hits == size for size, hits in self.ship_hits.items())

    def play_game(self, battleship_game):
        while not self.is_game_over():
            entropy_grid = self.calculate_entropy()
            information_gain_grid = self.calculate_information_gain()
            guess = self.make_guess(entropy_grid)
            result = 'hit' if battleship_game.check_hit(*guess) else 'miss'
            self.update_grid(*guess, result)
            print(f"Guess: {guess}, Result: {result}")
            self.plot_grid(battleship_game, information_gain_grid)  # Plot all three grids after each guess
    def animate_game(self, battleship_game):
        # Prepare the plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        plt.close()  # Prevents initial display of the empty plot

        # Animation update function
        def update(frame):
            nonlocal battleship_game
            if not self.is_game_over():
                entropy_grid = self.calculate_entropy()
                information_gain_grid = self.calculate_information_gain()
                guess = self.make_guess(entropy_grid)
                result = 'hit' if battleship_game.check_hit(*guess) else 'miss'
                self.update_grid(*guess, result)
                print(f"Guess: {guess}, Result: {result}")
            
            # Update each subplot
            sns.heatmap(self.grid, annot=True, cmap="coolwarm", cbar=False, fmt="d", ax=axes[0])
            sns.heatmap(battleship_game.grid, annot=True, cmap="YlGn", cbar=False, ax=axes[1])
            sns.heatmap(information_gain_grid, annot=True, cmap="viridis", cbar=True, ax=axes[2])

            for ax in axes:
                ax.clear()
            
            # Titles for subplots
            axes[0].set_title("Solver's Guesses (Hits/Misses)")
            axes[1].set_title("Actual Ship Placements")
            axes[2].set_title("Expected Information Gain")

        # Create the animation
        anim = animation.FuncAnimation(fig, update, frames=np.arange(100), repeat=False)
        return anim
class BayesianBattleshipSolver:
    def __init__(self, grid_size=10, ships=None):
        self.grid_size = grid_size
        self.grid = np.full((grid_size, grid_size), -1)  # -1 for unknown, 0 for miss, 1 for hit
        self.probability_grid = np.full((grid_size, grid_size), 1.0 / (grid_size * grid_size))
        self.ships = ships if ships else [5, 4, 3, 3, 2]

    def update_probabilities(self, guess, result):
        row, col = guess
        if result == 'hit':
            # Increase probability around the hit area
            self.update_adjacent_cells(row, col, increase=True)
        else:
            # Decrease probability of adjacent cells
            self.update_adjacent_cells(row, col, increase=False)
        # Normalize probabilities
        self.probability_grid /= np.sum(self.probability_grid)

    def update_adjacent_cells(self, row, col, increase):
        # Implement logic to update probabilities of adjacent cells
        pass

    def make_guess(self):
        # Choose the cell with the highest probability
        return np.unravel_index(np.argmax(self.probability_grid, axis=None), self.probability_grid.shape)

    def plot_probability_grid(self):
        plt.figure(figsize=(10, 6))
        sns.heatmap(self.probability_grid, annot=True, cmap="YlGnBu", cbar=True)
        plt.title("Probability Grid")
        plt.xlabel("Column")
        plt.ylabel("Row")
        plt.show()

    # Other methods similar to BattleshipSolver...

# Example usage
battleship_game = BattleshipGame()
solver = BattleshipSolver()
solver.play_game(battleship_game)

# from IPython.display import HTML
# anim = solver.animate_game(battleship_game)
# HTML(anim.to_jshtml())
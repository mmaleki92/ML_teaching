import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
LINE_COLOR = (0, 0, 255)

# Screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Confusion Matrix Demo')

# Fonts
font = pygame.font.SysFont(None, 28)

# Data set generator functions
def generate_clustered_data():
    """Two clearly separated clusters of virus and ok files."""
    virus_points = [(random.randint(50, WIDTH // 3), random.randint(50, HEIGHT - 50), 1) for _ in range(30)]  # Virus points on the left
    ok_points = [(random.randint(2 * WIDTH // 3, WIDTH - 50), random.randint(50, HEIGHT - 50), 0) for _ in range(30)]  # Ok files on the right
    return virus_points + ok_points

def generate_overlapping_data():
    """Overlapping clusters to simulate more difficult classification."""
    virus_points = [(random.randint(100, 400), random.randint(50, HEIGHT - 50), 1) for _ in range(30)]
    ok_points = [(random.randint(200, 700), random.randint(50, HEIGHT - 50), 0) for _ in range(30)]
    return virus_points + ok_points

def generate_circular_data():
    """Circular distribution of points."""
    center = (WIDTH // 2, HEIGHT // 2)
    radius = 150
    
    virus_points = []
    ok_points = []
    
    for _ in range(30):
        angle = random.uniform(0, 2 * math.pi)
        dist = random.randint(0, radius)
        x = center[0] + int(dist * math.cos(angle))
        y = center[1] + int(dist * math.sin(angle))
        virus_points.append((x, y, 1))  # Virus inside the circle
        
    for _ in range(30):
        angle = random.uniform(0, 2 * math.pi)
        dist = random.randint(radius + 50, radius + 150)
        x = center[0] + int(dist * math.cos(angle))
        y = center[1] + int(dist * math.sin(angle))
        ok_points.append((x, y, 0))  # Ok outside the circle

    return virus_points + ok_points

def generate_diagonal_data():
    """Points distributed diagonally."""
    virus_points = [(random.randint(50, WIDTH // 2 - 50), random.randint(50, HEIGHT // 2 - 50), 1) for _ in range(30)]
    ok_points = [(random.randint(WIDTH // 2 + 50, WIDTH - 50), random.randint(HEIGHT // 2 + 50, HEIGHT - 50), 0) for _ in range(30)]
    return virus_points + ok_points

# List of datasets
datasets = [generate_clustered_data, generate_overlapping_data, generate_circular_data, generate_diagonal_data]
dataset_index = 0
points = datasets[dataset_index]()

# Movable line
line_x = WIDTH // 2

# Function to calculate confusion matrix metrics
def calculate_metrics(points, line_x):
    TP = TN = FP = FN = 0
    for (x, y, label) in points:
        predicted = 1 if x < line_x else 0  # Left side of line = virus (1)
        
        if label == 1 and predicted == 1:
            TP += 1
        elif label == 0 and predicted == 0:
            TN += 1
        elif label == 0 and predicted == 1:
            FP += 1
        elif label == 1 and predicted == 0:
            FN += 1
    
    # Calculating metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return TP, TN, FP, FN, accuracy, recall, precision, f1_score

# Function to display metrics
def display_metrics(metrics):
    TP, TN, FP, FN, accuracy, recall, precision, f1_score = metrics
    metrics_text = [
        f"True Positives: {TP}",
        f"True Negatives: {TN}",
        f"False Positives: {FP}",
        f"False Negatives: {FN}",
        f"Accuracy: {accuracy:.2f}",
        f"Recall: {recall:.2f}",
        f"Precision: {precision:.2f}",
        f"F1 Score: {f1_score:.2f}"
    ]
    
    for i, text in enumerate(metrics_text):
        msg = font.render(text, True, BLACK)
        screen.blit(msg, (10, 10 + i * 30))

# Main game loop
running = True
while running:
    screen.fill(WHITE)
    
    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEMOTION:
            line_x = event.pos[0]
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                # Switch to the next dataset on spacebar press
                dataset_index = (dataset_index + 1) % len(datasets)
                points = datasets[dataset_index]()

    # Draw points
    for (x, y, label) in points:
        color = RED if label == 1 else GREEN
        pygame.draw.circle(screen, color, (x, y), 5)

    # Draw line
    pygame.draw.line(screen, LINE_COLOR, (line_x, 0), (line_x, HEIGHT), 3)

    # Calculate and display metrics
    metrics = calculate_metrics(points, line_x)
    display_metrics(metrics)

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()

# Example 1: Basic Linear Regression

Objective:

Implement a simple linear regression model using PyTorch. Learn the basic workflow of defining a model, loss function, optimizer, and training loop.

```python

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data
X = np.random.rand(100, 1) * 10
y = 2 * X + 1 + np.random.randn(100, 1) * 1.5  # y = 2x + 1 + noise

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Define the model
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)  # input_dim = 1, output_dim = 1
        
    def forward(self, x):
        return self.linear(x)

# Initialize the model
model = LinearRegressionModel()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)
    losses.append(loss.item())
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Print model parameters
print(f'Weight: {model.linear.weight.item():.4f}, Bias: {model.linear.bias.item():.4f}')

# Visualize results
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
with torch.no_grad():
    plt.scatter(X, y, label='Original data')
    plt.plot(X, model(X_tensor).numpy(), color='red', label='Fitted line')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression Result')
plt.legend()
plt.tight_layout()
plt.show()

```

# Example 2: Binary Classification with Logistic Regression

Objective:

Implement a simple binary classification model using logistic regression. Learn how to use the sigmoid activation function and binary cross-entropy loss.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate two-class spiral data
def generate_spiral_data(samples_per_class=100, noise=0.1):
    points_per_class = samples_per_class
    X = np.zeros((points_per_class*2, 2))
    y = np.zeros(points_per_class*2, dtype='uint8')
    
    for class_idx in range(2):
        r = np.linspace(0.01, 1, points_per_class)
        t = np.linspace(class_idx*4, (class_idx+1)*4, points_per_class) + np.random.randn(points_per_class)*noise
        ix = range(points_per_class*class_idx, points_per_class*(class_idx+1))
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = class_idx
    
    return X, y

# Generate the data
X, y = generate_spiral_data()
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).unsqueeze(1)  # Add dimension for BCE loss

# Define the model
class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 10)
        self.layer2 = nn.Linear(10, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.sigmoid(self.layer2(x))
        return x

# Initialize the model
model = BinaryClassifier()

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)
    losses.append(loss.item())
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Calculate accuracy
with torch.no_grad():
    y_pred_class = (model(X_tensor) > 0.5).float()
    accuracy = (y_pred_class == y_tensor).float().mean()
    print(f'Accuracy: {accuracy.item():.4f}')

# Visualize decision boundary
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
# Create a grid to evaluate model
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

with torch.no_grad():
    Z = model(grid).reshape(xx.shape)
    
plt.contourf(xx, yy, Z.numpy(), alpha=0.3, levels=np.linspace(0, 1, 3))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.tight_layout()
plt.show()
```


# Example 3: Multi-class Classification

Objective:

Implement a neural network for multi-class classification using the Softmax activation function and CrossEntropyLoss.


```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate multi-class synthetic data
def generate_multiclass_data(num_classes=3, samples_per_class=100, noise=0.1):
    X = np.zeros((samples_per_class*num_classes, 2))
    y = np.zeros(samples_per_class*num_classes, dtype='int64')
    
    for class_idx in range(num_classes):
        r = np.linspace(0.01, 1, samples_per_class)
        t = np.linspace(class_idx*(2*np.pi/num_classes), (class_idx+1)*(2*np.pi/num_classes), samples_per_class) + np.random.randn(samples_per_class)*noise
        ix = range(samples_per_class*class_idx, samples_per_class*(class_idx+1))
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = class_idx
    
    return X, y

# Generate data
X, y = generate_multiclass_data(num_classes=3)
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)  # LongTensor for CrossEntropyLoss

# Define the model
class MultiClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.layer1 = nn.Linear(2, 20)
        self.layer2 = nn.Linear(20, 20)
        self.layer3 = nn.Linear(20, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.layer3(x)  # No softmax here - CrossEntropyLoss includes it
        return x

# Initialize the model
model = MultiClassifier()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # Includes softmax internally
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)
    losses.append(loss.item())
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Calculate accuracy
with torch.no_grad():
    _, y_pred_class = torch.max(model(X_tensor), 1)
    accuracy = (y_pred_class == y_tensor).float().mean()
    print(f'Accuracy: {accuracy.item():.4f}')

# Visualize decision boundaries
plt.figure(figsize=(10, 6))
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

plt.subplot(1, 2, 2)
# Create a grid to evaluate model
x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))
grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])

with torch.no_grad():
    Z = torch.argmax(model(grid), dim=1).reshape(xx.shape)
    
plt.contourf(xx, yy, Z.numpy(), alpha=0.3, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Multi-class Decision Boundaries')
plt.tight_layout()
plt.show()

```

# Example 4: Visualizing Model Weights

Objective:

Learn how to access and visualize the weights of a trained neural network. This helps in understanding what the model has learned.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate simple MNIST-like data (simplified to 5x5 digits)
def generate_simple_digit_data(num_samples=100):
    # Create templates for digits 0 and 1 (5x5 pixels)
    digit_0 = np.array([
        [0, 1, 1, 1, 0],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [0, 1, 1, 1, 0]
    ])
    
    digit_1 = np.array([
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0]
    ])
    
    X = np.zeros((num_samples, 25))  # Flattened 5x5 images
    y = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        if i % 2 == 0:  # Even indices are digit 0
            digit = digit_0.copy()
            y[i] = 0
        else:  # Odd indices are digit 1
            digit = digit_1.copy()
            y[i] = 1
            
        # Add noise
        noise = np.random.normal(0, 0.1, digit.shape)
        digit = np.clip(digit + noise, 0, 1)
        X[i] = digit.flatten()
    
    return X, y

# Generate data
X, y = generate_simple_digit_data(200)
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# Define the model
class SimpleNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(25, 10)  # First layer (we'll visualize these weights)
        self.fc2 = nn.Linear(10, 2)   # Output layer (2 classes)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Initialize the model
model = SimpleNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Calculate accuracy
with torch.no_grad():
    _, predicted = torch.max(model(X_tensor), 1)
    accuracy = (predicted == y_tensor).float().mean()
    print(f'Accuracy: {accuracy.item():.4f}')

# Visualize the weights of the first layer
plt.figure(figsize=(15, 6))

# Plot original example digits
plt.subplot(2, 5, 1)
plt.imshow(X[0].reshape(5, 5), cmap='gray')
plt.title('Sample Digit 0')
plt.axis('off')

plt.subplot(2, 5, 2)
plt.imshow(X[1].reshape(5, 5), cmap='gray')
plt.title('Sample Digit 1')
plt.axis('off')

# Get first layer weights
weights = model.fc1.weight.data.numpy()

# Plot the weights for each neuron in the first hidden layer
for i in range(8):  # Show 8 of the 10 neurons
    plt.subplot(2, 5, i+3)
    plt.imshow(weights[i].reshape(5, 5), cmap='viridis')
    plt.title(f'Neuron {i+1} Weights')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Also visualize the activations for a sample input
plt.figure(figsize=(10, 4))

# Take a sample input
sample_input = X_tensor[0]  # Sample digit 0
sample_input_reshaped = sample_input.reshape(5, 5)

# Get activations from the first layer
with torch.no_grad():
    first_layer_output = model.relu(model.fc1(sample_input))

plt.subplot(1, 2, 1)
plt.imshow(sample_input_reshaped, cmap='gray')
plt.title('Input Digit')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.bar(range(10), first_layer_output.numpy())
plt.title('Activation of First Layer Neurons')
plt.xlabel('Neuron Index')
plt.ylabel('Activation')
plt.tight_layout()
plt.show()
```


# Example 5: Simple Feedforward Neural Network for Regression

Objective:

Implement a feedforward neural network with multiple hidden layers for a regression task. Learn how to use non-linear activation functions and build deeper networks.


```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate non-linear regression data
x = np.random.uniform(-3, 3, 200)
y = np.sin(x) * np.exp(-0.1 * abs(x)) + np.random.normal(0, 0.1, size=x.shape)

# Reshape and convert to tensors
X = x.reshape(-1, 1)
Y = y.reshape(-1, 1)
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(Y)

# Define the model
class FeedForwardNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(1, 32)
        self.layer2 = nn.Linear(32, 64)
        self.layer3 = nn.Linear(64, 32)
        self.layer4 = nn.Linear(32, 1)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.tanh(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        x = self.layer4(x)
        return x

# Initialize the model
model = FeedForwardNN()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 1000
losses = []

for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_tensor)
    loss = criterion(y_pred, y_tensor)
    losses.append(loss.item())
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Visualize results
plt.figure(figsize=(15, 5))

# Plot loss curve
plt.subplot(1, 3, 1)
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# Plot the predictions vs actual
plt.subplot(1, 3, 2)
with torch.no_grad():
    # Sort points for better visualization
    sorted_indices = np.argsort(x)
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Make predictions
    X_tensor_sorted = torch.FloatTensor(x_sorted.reshape(-1, 1))
    predictions = model(X_tensor_sorted).numpy().flatten()
    
plt.scatter(x, y, color='blue', alpha=0.5, label='Original data')
plt.plot(x_sorted, predictions, color='red', linewidth=2, label='Model prediction')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Predictions vs Actual')
plt.legend()

# Visualize each layer's activations for a range of inputs
plt.subplot(1, 3, 3)
test_x = np.linspace(-3, 3, 100).reshape(-1, 1)
test_x_tensor = torch.FloatTensor(test_x)

with torch.no_grad():
    layer1_out = model.tanh(model.layer1(test_x_tensor)).numpy()
    layer2_out = model.relu(model.layer2(model.tanh(model.layer1(test_x_tensor)))).numpy()
    
# Plot activations from the first hidden layer
plt.plot(test_x, layer1_out[:, 0], label='Layer 1, Unit 0')
plt.plot(test_x, layer1_out[:, 1], label='Layer 1, Unit 1')
plt.plot(test_x, layer1_out[:, 2], label='Layer 1, Unit 2')
plt.plot(test_x, layer2_out[:, 0], label='Layer 2, Unit 0')
plt.title('Layer Activations')
plt.xlabel('x')
plt.ylabel('Activation')
plt.legend()

plt.tight_layout()
plt.show()
```

# Example 6: Binary Classification with Imbalanced Data

Objective:

Learn how to handle imbalanced datasets for binary classification using weighted loss functions. This is a common challenge in real-world machine learning problems.

```python

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate imbalanced binary classification data
def generate_imbalanced_data(n_samples=1000, imbalance_ratio=0.1):
    # Class 0 (majority class)
    n_majority = int(n_samples * (1 - imbalance_ratio))
    X_majority = np.random.randn(n_majority, 2) * 1.5
    
    # Class 1 (minority class)
    n_minority = n_samples - n_majority
    X_minority = np.random.randn(n_minority, 2) * 0.5 + np.array([2.0, 2.0])
    
    # Combine data
    X = np.vstack([X_majority, X_minority])
    y = np.hstack([np.zeros(n_majority), np.ones(n_minority)])
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    return X, y

# Generate imbalanced data
X, y = generate_imbalanced_data(imbalance_ratio=0.1)
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y).unsqueeze(1)

# Calculate class weights for weighted loss
n_samples = len(y)
n_class_1 = sum(y)
n_class_0 = n_samples - n_class_1
weight_class_0 = n_samples / (2 * n_class_0)
weight_class_1 = n_samples / (2 * n_class_1)
pos_weight = torch.FloatTensor([weight_class_1 / weight_class_0])

# Define the model
class BinaryClassifierImbalanced(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 8)
        self.layer3 = nn.Linear(8, 1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))
        return x

# Initialize the model
model = BinaryClassifierImbalanced()

# Define weighted loss function and optimizer
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Using BCEWithLogitsLoss for numerical stability
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 500
losses = []

for epoch in range(num_epochs):
    # Forward pass (without sigmoid since it's included in BCEWithLogitsLoss)
    logits = model.layer3(model.relu(model.layer2(model.relu(model.layer1(X_tensor)))))
    loss = criterion(logits, y_tensor)
    losses.append(loss.item())
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 100 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

# Evaluate the model
with torch.no_grad():
    # Apply sigmoid to get probabilities
    y_pred_prob = torch.sigmoid(model.layer3(model.relu(model.layer2(model.relu(model.layer1(X_tensor))))))
    y_pred = (y_pred_prob > 0.5).float()
    
    # Calculate metrics
    accuracy = (y_pred == y_tensor).float().mean().item()
    precision, recall, f1, _ = precision_recall_fscore_support(y, y_pred.numpy(), average='binary')
    
    # Confusion matrix
    cm = confusion_matrix(y, y_pred.numpy())
    
    print(f'Accuracy: {accuracy:.4f}')
    print(f'Precision: {precision:.4f}')
    print(f'Recall: {recall:.4f}')
    print(f'F1 Score: {f1:.4f}')
    print(f'Confusion Matrix:\n{cm}')

# Visualize results
plt.figure(figsize=(15, 5))

# Plot training loss
plt.subplot(1, 3, 1)
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# Plot decision boundary
plt.subplot(1, 3, 2)
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

with torch.no_grad():
    grid = torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])
    logits = model.layer3(model.relu(model.layer2(model.relu(model.layer1(grid)))))
    probs = torch.sigmoid(logits).reshape(xx.shape)
    
plt.contourf(xx, yy, probs.numpy(), alpha=0.3, cmap=plt.cm.Spectral)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')

# Plot confusion matrix
plt.subplot(1, 3, 3)
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()
plt.xticks([0, 1], ['Class 0', 'Class 1'])
plt.yticks([0, 1], ['Class 0', 'Class 1'])

plt.tight_layout()
plt.show()

```

# Example 7: Simple CNN for Image Classification

Objective:

Implement a basic Convolutional Neural Network (CNN) for image classification. Learn about convolutional layers, pooling, and flattening operations.


```python

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic image data (simplified MNIST-like)
def generate_image_data(num_samples=500, img_size=28, num_classes=3):
    X = np.zeros((num_samples, 1, img_size, img_size))
    y = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        # Randomly assign a class
        class_idx = i % num_classes
        y[i] = class_idx
        
        # Create a blank image
        img = np.zeros((img_size, img_size))
        
        # Draw different shapes based on class
        if class_idx == 0:  # Class 0: Circle
            center = img_size // 2
            radius = img_size // 4
            for x in range(img_size):
                for y in range(img_size):
                    if ((x - center) ** 2 + (y - center) ** 2) < radius ** 2:
                        img[x, y] = 1.0
                        
        elif class_idx == 1:  # Class 1: Square
            start = img_size // 4
            end = 3 * img_size // 4
            img[start:end, start:end] = 1.0
            
        elif class_idx == 2:  # Class 2: Cross
            thickness = img_size // 8
            center = img_size // 2
            
            # Horizontal line
            img[center-thickness//2:center+thickness//2, :] = 1.0
            # Vertical line
            img[:, center-thickness//2:center+thickness//2] = 1.0
        
        # Add noise
        img += np.random.normal(0, 0.1, img.shape)
        img = np.clip(img, 0, 1)
        
        X[i, 0] = img
    
    # Shuffle the data
    idx = np.random.permutation(num_samples)
    X, y = X[idx], y[idx]
    
    return X, y

# Generate data
X, y = generate_image_data(num_samples=500, img_size=28, num_classes=3)
X_tensor = torch.FloatTensor(X)
y_tensor = torch.LongTensor(y)

# Define the CNN model
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)  # 28/2/2 = 7
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, num_classes)
        
    def forward(self, x):
        # First conv block
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        # Flatten and fully connected layers
        x = x.view(-1, 32 * 7 * 7)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        
        return x
    
    def feature_maps(self, x):
        # Get feature maps from first convolutional layer
        return self.conv1(x)

# Initialize the model
model = SimpleCNN()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split data into train and test
train_size = int(0.8 * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Training loop
num_epochs = 10
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    y_pred = model(X_train)
    train_loss = criterion(y_pred, y_train)
    train_losses.append(train_loss.item())
    
    # Backward pass and optimize
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    # Evaluation phase
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)
        test_losses.append(test_loss.item())
        
        # Calculate accuracy
        _, predicted = torch.max(y_pred, 1)
        accuracy = (predicted == y_test).float().mean()
        test_accuracies.append(accuracy.item())
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Accuracy: {accuracy.item():.4f}')

# Visualize results
plt.figure(figsize=(15, 10))

# Plot the training and test loss
plt.subplot(2, 3, 1)
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

# Plot test accuracy
plt.subplot(2, 3, 2)
plt.plot(range(num_epochs), test_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')

# Visualize sample images
plt.subplot(2, 3, 3)
grid_size = 3
for i in range(grid_size):
    for j in range(grid_size):
        idx = i * grid_size + j
        plt.subplot(2, 3, 3 + idx)
        plt.imshow(X[idx, 0], cmap='gray')
        plt.title(f'Class: {y[idx]}')
        plt.axis('off')

# Visualize feature maps for one sample image
plt.figure(figsize=(15, 5))
sample_img = X_tensor[0:1]  # Add batch dimension
feature_maps = model.feature_maps(sample_img).detach().numpy()[0]

plt.subplot(1, (feature_maps.shape[0] // 4) + 2, 1)
plt.imshow(X[0, 0], cmap='gray')
plt.title('Original Image')
plt.axis('off')

for i in range(min(16, feature_maps.shape[0])):
    plt.subplot(1, (feature_maps.shape[0] // 4) + 2, i+2)
    plt.imshow(feature_maps[i], cmap='viridis')
    plt.title(f'Feature Map {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Visualize convolutional kernels
plt.figure(figsize=(12, 5))
conv1_weights = model.conv1.weight.data.numpy()

for i in range(min(16, conv1_weights.shape[0])):
    plt.subplot(4, 4, i+1)
    plt.imshow(conv1_weights[i, 0], cmap='viridis')
    plt.title(f'Kernel {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.suptitle('First Convolutional Layer Kernels', y=1.05)
plt.show()

```


# Example 8: Multi-label Classification


Objective:

Implement a neural network for multi-label classification where each sample can belong to multiple classes simultaneously. Learn how to handle multiple outputs and use the Binary Cross Entropy loss.

```python

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import hamming_loss, jaccard_score

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate multi-label data
def generate_multilabel_data(num_samples=1000, num_features=20, num_labels=5):
    X = np.random.randn(num_samples, num_features)
    
    # Create label correlations for realism
    W = np.random.randn(num_features, num_labels) * 0.5
    
    # Generate raw scores and apply sigmoid
    scores = np.dot(X, W)
    y_prob = 1 / (1 + np.exp(-scores))
    
    # Convert to binary labels
    y = (y_prob > 0.5).astype(np.float32)
    
    return X, y

# Generate data
X, y = generate_multilabel_data(num_samples=1000, num_features=20, num_labels=5)
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# Define the model
class MultilabelClassifier(nn.Module):
    def __init__(self, input_dim=20, num_labels=5):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, num_labels)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.sigmoid(self.layer3(x))  # Sigmoid for multilabel output
        return x

# Initialize the model
model = MultilabelClassifier()

# Define loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy for multilabel classification
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split data into train and test
train_size = int(0.8 * len(X))
X_train, X_test = X_tensor[:train_size], X_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Training loop
num_epochs = 50
train_losses = []
test_losses = []
hamming_scores = []
jaccard_scores = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    y_pred = model(X_train)
    train_loss = criterion(y_pred, y_train)
    train_losses.append(train_loss.item())
    
    # Backward pass and optimize
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    # Evaluation phase
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        test_loss = criterion(y_pred, y_test)
        test_losses.append(test_loss.item())
        
        # Convert probabilistic predictions to binary
        y_pred_binary = (y_pred > 0.5).float()
        
        # Calculate metrics
        hamming = hamming_loss(y_test.numpy(), y_pred_binary.numpy())
        hamming_scores.append(hamming)
        
        jaccard = jaccard_score(y_test.numpy(), y_pred_binary.numpy(), average='samples')
        jaccard_scores.append(jaccard)
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}')
        print(f'Hamming Loss: {hamming:.4f}, Jaccard Score: {jaccard:.4f}')

# Visualize results
plt.figure(figsize=(15, 10))

# Plot the training and test loss
plt.subplot(2, 2, 1)
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

# Plot Hamming Loss (lower is better)
plt.subplot(2, 2, 2)
plt.plot(range(num_epochs), hamming_scores)
plt.xlabel('Epoch')
plt.ylabel('Hamming Loss')
plt.title('Hamming Loss (lower is better)')

# Plot Jaccard Similarity (higher is better)
plt.subplot(2, 2, 3)
plt.plot(range(num_epochs), jaccard_scores)
plt.xlabel('Epoch')
plt.ylabel('Jaccard Score')
plt.title('Jaccard Similarity (higher is better)')

# Visualize label co-occurrences
plt.subplot(2, 2, 4)
co_occurrence = y.T @ y  # Matrix multiplication gives co-occurrence counts
plt.imshow(co_occurrence, cmap='viridis')
plt.colorbar()
plt.title('Label Co-occurrence Matrix')
plt.xlabel('Label Index')
plt.ylabel('Label Index')

plt.tight_layout()
plt.show()

# Visualize model predictions for a few samples
plt.figure(figsize=(12, 6))
num_samples_to_show = 5

with torch.no_grad():
    y_pred = model(X_test[:num_samples_to_show])

for i in range(num_samples_to_show):
    plt.subplot(2, num_samples_to_show, i+1)
    plt.bar(range(y_test.shape[1]), y_test[i].numpy())
    plt.title(f'True Labels - Sample {i+1}')
    plt.ylim(0, 1)
    plt.xticks(range(y_test.shape[1]))
    
    plt.subplot(2, num_samples_to_show, i+1+num_samples_to_show)
    plt.bar(range(y_pred.shape[1]), y_pred[i].numpy())
    plt.title(f'Predicted Probabilities - Sample {i+1}')
    plt.ylim(0, 1)
    plt.xticks(range(y_pred.shape[1]))

plt.tight_layout()
plt.show()

```

# Example 9: Simple Autoencoder

Objective:

Implement a simple autoencoder to learn compressed representations of data. Learn about encoder-decoder architectures and reconstruction loss.

```python

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic data with underlying patterns
def generate_autoencoder_data(num_samples=1000, input_dim=20, latent_dim=2):
    # Generate data along a manifold
    t = np.random.uniform(0, 2*np.pi, num_samples)
    latent_factors = np.column_stack([np.sin(t), np.cos(t)])  # Circle in 2D
    
    # Create a random projection matrix to higher dimensions
    projection = np.random.randn(latent_dim, input_dim) * 0.1
    
    # Project the latent factors to higher dimensions
    X = latent_factors @ projection
    
    # Add noise
    X += np.random.normal(0, 0.1, X.shape)
    
    return X, latent_factors

# Generate data
X, latent_factors = generate_autoencoder_data(num_samples=1000, input_dim=20, latent_dim=2)
X_tensor = torch.FloatTensor(X)

# Define the autoencoder model
class SimpleAutoencoder(nn.Module):
    def __init__(self, input_dim=20, latent_dim=2):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        # Encode the input
        z = self.encoder(x)
        # Decode the latent representation
        reconstructed = self.decoder(z)
        return reconstructed, z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

# Initialize the model
model = SimpleAutoencoder(input_dim=20, latent_dim=2)

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    # Forward pass
    reconstructed, latent = model(X_tensor)
    loss = criterion(reconstructed, X_tensor)
    losses.append(loss.item())
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.6f}')

# Extract the learned latent space
with torch.no_grad():
    _, learned_latent = model(X_tensor)
    learned_latent = learned_latent.numpy()

# Perform PCA for comparison
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X)

# Visualize results
plt.figure(figsize=(15, 10))

# Plot the training loss
plt.subplot(2, 2, 1)
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# Plot the true latent factors
plt.subplot(2, 2, 2)
plt.scatter(latent_factors[:, 0], latent_factors[:, 1], alpha=0.5, c=t, cmap='viridis')
plt.colorbar(label='Parameter t')
plt.title('True Latent Factors')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# Plot the learned latent space
plt.subplot(2, 2, 3)
plt.scatter(learned_latent[:, 0], learned_latent[:, 1], alpha=0.5, c=t, cmap='viridis')
plt.colorbar(label='Parameter t')
plt.title('Learned Latent Space')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# Plot the PCA result
plt.subplot(2, 2, 4)
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5, c=t, cmap='viridis')
plt.colorbar(label='Parameter t')
plt.title('PCA Result')
plt.xlabel('PC 1')
plt.ylabel('PC 2')

plt.tight_layout()
plt.show()

# Experiment with the latent space
plt.figure(figsize=(15, 5))

# Generate points in the latent space
grid_size = 20
x = np.linspace(-2, 2, grid_size)
y = np.linspace(-2, 2, grid_size)
z_grid = np.array([(i, j) for i in x for j in y])
z_tensor = torch.FloatTensor(z_grid)

# Decode the latent points
with torch.no_grad():
    decoded = model.decode(z_tensor).numpy()

# Perform PCA on the decoded points to visualize
decoded_pca = pca.transform(decoded)

# Plot the latent space grid
plt.subplot(1, 2, 1)
plt.scatter(z_grid[:, 0], z_grid[:, 1], alpha=0.3)
plt.title('Regular Grid in Latent Space')
plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')

# Plot the decoded points
plt.subplot(1, 2, 2)
plt.scatter(decoded_pca[:, 0], decoded_pca[:, 1], alpha=0.3)
plt.title('Decoded Grid (projected to 2D with PCA)')
plt.xlabel('PC 1')
plt.ylabel('PC 2')

plt.tight_layout()
plt.show()

```

# Example 10: Image Autoencoder with Convolutional Layers

Objective:

Implement a convolutional autoencoder for image data. Learn about upsampling, transpose convolutions, and how to visualize the reconstructed images.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic image data
def generate_synthetic_images(num_samples=500, img_size=28):
    images = np.zeros((num_samples, 1, img_size, img_size))
    
    for i in range(num_samples):
        # Create a blank image
        img = np.zeros((img_size, img_size))
        
        # Randomly determine the shape and parameters
        shape_type = np.random.randint(0, 3)
        center_x = np.random.randint(img_size // 4, 3 * img_size // 4)
        center_y = np.random.randint(img_size // 4, 3 * img_size // 4)
        size = np.random.randint(img_size // 6, img_size // 3)
        
        if shape_type == 0:  # Circle
            for x in range(img_size):
                for y in range(img_size):
                    if ((x - center_x) ** 2 + (y - center_y) ** 2) < size ** 2:
                        img[x, y] = 1.0
        
        elif shape_type == 1:  # Square
            x_start = max(0, center_x - size)
            x_end = min(img_size, center_x + size)
            y_start = max(0, center_y - size)
            y_end = min(img_size, center_y + size)
            img[x_start:x_end, y_start:y_end] = 1.0
        
        else:  # Triangle
            points = np.array([
                [center_x, center_y - size],
                [center_x - size, center_y + size],
                [center_x + size, center_y + size]
            ])
            
            # Simple triangle filling algorithm
            x_min, y_min = np.min(points, axis=0)
            x_max, y_max = np.max(points, axis=0)
            
            for x in range(max(0, int(x_min)), min(img_size, int(x_max) + 1)):
                for y in range(max(0, int(y_min)), min(img_size, int(y_max) + 1)):
                    # Check if point is inside triangle using barycentric coordinates
                    p = np.array([x, y])
                    v0 = points[1] - points[0]
                    v1 = points[2] - points[0]
                    v2 = p - points[0]
                    
                    dot00 = np.dot(v0, v0)
                    dot01 = np.dot(v0, v1)
                    dot02 = np.dot(v0, v2)
                    dot11 = np.dot(v1, v1)
                    dot12 = np.dot(v1, v2)
                    
                    invDenom = 1 / (dot00 * dot11 - dot01 * dot01)
                    u = (dot11 * dot02 - dot01 * dot12) * invDenom
                    v = (dot00 * dot12 - dot01 * dot02) * invDenom
                    
                    if (u >= 0) and (v >= 0) and (u + v <= 1):
                        img[x, y] = 1.0
        
        # Add noise
        img += np.random.normal(0, 0.1, img.shape)
        img = np.clip(img, 0, 1)
        
        images[i, 0] = img
    
    return images

# Generate data
images = generate_synthetic_images(num_samples=500, img_size=28)
images_tensor = torch.FloatTensor(images)

# Define the convolutional autoencoder model
class ConvAutoencoder(nn.Module):
    def __init__(self, latent_dim=8):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 28x28 -> 14x14
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 14x14 -> 7x7
            nn.Flatten(),
            nn.Linear(8 * 7 * 7, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 8 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (8, 7, 7)),
            nn.ConvTranspose2d(8, 16, kernel_size=4, stride=2, padding=1),  # 7x7 -> 14x14
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2, padding=1),  # 14x14 -> 28x28
            nn.Sigmoid()
        )
    
    def forward(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, z):
        return self.decoder(z)

# Initialize the model
model = ConvAutoencoder(latent_dim=8)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split data into train and test
train_size = int(0.8 * len(images))
train_images = images_tensor[:train_size]
test_images = images_tensor[train_size:]

# Training loop
num_epochs = 50
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # Train
    model.train()
    reconstructed, _ = model(train_images)
    train_loss = criterion(reconstructed, train_images)
    train_losses.append(train_loss.item())
    
    # Backward pass and optimize
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        reconstructed_test, _ = model(test_images)
        test_loss = criterion(reconstructed_test, test_images)
        test_losses.append(test_loss.item())
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item():.6f}, Test Loss: {test_loss.item():.6f}')

# Visualize results
plt.figure(figsize=(15, 10))

# Plot the training and test loss
plt.subplot(2, 3, 1)
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

# Visualize reconstructions
with torch.no_grad():
    reconstructed_test, latent_test = model(test_images[:10])

for i in range(5):
    # Original image
    plt.subplot(2, 5, i+1+1)
    plt.imshow(test_images[i, 0].numpy(), cmap='gray')
    plt.title(f'Original {i+1}')
    plt.axis('off')
    
    # Reconstructed image
    plt.subplot(2, 5, i+6+1)
    plt.imshow(reconstructed_test[i, 0].numpy(), cmap='gray')
    plt.title(f'Reconstructed {i+1}')
    plt.axis('off')

plt.tight_layout()
plt.show()

# Visualize the latent space with PCA
with torch.no_grad():
    _, latent_vectors = model(images_tensor)
    latent_vectors = latent_vectors.numpy()

# Apply PCA to latent vectors
pca = PCA(n_components=2)
latent_pca = pca.fit_transform(latent_vectors)

plt.figure(figsize=(8, 6))
plt.scatter(latent_pca[:, 0], latent_pca[:, 1], alpha=0.5, cmap='viridis')
plt.title('PCA of Latent Space Vectors')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar()
plt.tight_layout()
plt.show()

# Generate new images from the latent space
plt.figure(figsize=(12, 12))

# Create a grid of points in the 2D PCA space
grid_size = 5
x_min, x_max = latent_pca[:, 0].min(), latent_pca[:, 0].max()
y_min, y_max = latent_pca[:, 1].min(), latent_pca[:, 1].max()
x_grid = np.linspace(x_min, x_max, grid_size)
y_grid = np.linspace(y_min, y_max, grid_size)

# For each grid point, find the closest latent vector
# For each grid point, find the closest latent vector
for i, y_val in enumerate(y_grid):
    for j, x_val in enumerate(x_grid):
        # Find the closest latent vector in PCA space
        dists = np.sum((latent_pca - np.array([x_val, y_val])) ** 2, axis=1)
        closest_idx = np.argmin(dists)
        
        # Get the original latent vector (before PCA)
        z = latent_vectors[closest_idx]
        
        # Convert to tensor and generate image
        z_tensor = torch.FloatTensor(z).unsqueeze(0)
        
        with torch.no_grad():
            generated = model.decode(z_tensor)
        
        # Plot
        plt.subplot(grid_size, grid_size, i * grid_size + j + 1)
        plt.imshow(generated[0, 0].numpy(), cmap='gray')
        plt.axis('off')

plt.tight_layout()
plt.suptitle('Generated Images from Latent Space', y=1.02, fontsize=16)
plt.show()
```
# Example 15: Simple Neural Network with Multi-view Learning

Objective:

Implement a neural network that can handle inputs from multiple views (different representations of the same data). Learn about feature fusion and joint representation learning.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate multi-view synthetic data
def generate_multiview_data(num_samples=500, view1_dim=20, view2_dim=15, num_classes=3):
    # Generate class centers for each view
    centers_view1 = np.random.randn(num_classes, view1_dim) * 2
    centers_view2 = np.random.randn(num_classes, view2_dim) * 2
    
    X_view1 = np.zeros((num_samples, view1_dim))
    X_view2 = np.zeros((num_samples, view2_dim))
    y = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        # Assign a class
        class_idx = i % num_classes
        y[i] = class_idx
        
        # Generate data point for view 1
        X_view1[i] = centers_view1[class_idx] + np.random.randn(view1_dim) * 0.5
        
        # Generate data point for view 2 (different feature space)
        X_view2[i] = centers_view2[class_idx] + np.random.randn(view2_dim) * 0.5
    
    # Shuffle the data
    idx = np.random.permutation(num_samples)
    X_view1, X_view2, y = X_view1[idx], X_view2[idx], y[idx]
    
    return X_view1, X_view2, y

# Generate data
X_view1, X_view2, y = generate_multiview_data(num_samples=500, view1_dim=20, view2_dim=15, num_classes=3)
X_view1_tensor = torch.FloatTensor(X_view1)
X_view2_tensor = torch.FloatTensor(X_view2)
y_tensor = torch.LongTensor(y)

# Define the multi-view model
class MultiViewNetwork(nn.Module):
    def __init__(self, view1_dim=20, view2_dim=15, hidden_dim=32, num_classes=3):
        super().__init__()
        
        # View 1 encoder
        self.view1_encoder = nn.Sequential(
            nn.Linear(view1_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # View 2 encoder
        self.view2_encoder = nn.Sequential(
            nn.Linear(view2_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Joint representation and classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x1, x2):
        # Encode each view
        view1_encoded = self.view1_encoder(x1)
        view2_encoded = self.view2_encoder(x2)
        
        # Concatenate the encoded views
        joint_representation = torch.cat((view1_encoded, view2_encoded), dim=1)
        
        # Classify based on the joint representation
        output = self.classifier(joint_representation)
        
        return output, joint_representation

# Initialize the model
model = MultiViewNetwork()

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Split data into train and test
train_size = int(0.8 * len(y))
X1_train, X1_test = X_view1_tensor[:train_size], X_view1_tensor[train_size:]
X2_train, X2_test = X_view2_tensor[:train_size], X_view2_tensor[train_size:]
y_train, y_test = y_tensor[:train_size], y_tensor[train_size:]

# Training loop
num_epochs = 50
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(num_epochs):
    # Train
    model.train()
    outputs, _ = model(X1_train, X2_train)
    train_loss = criterion(outputs, y_train)
    train_losses.append(train_loss.item())
    
    # Backward pass and optimize
    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()
    
    # Evaluate
    model.eval()
    with torch.no_grad():
        outputs, _ = model(X1_test, X2_test)
        test_loss = criterion(outputs, y_test)
        test_losses.append(test_loss.item())
        
        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        accuracy = (predicted == y_test).float().mean()
        test_accuracies.append(accuracy.item())
    
    if (epoch+1) % 5 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss.item():.4f}, Test Loss: {test_loss.item():.4f}, Accuracy: {accuracy.item():.4f}')

# Evaluate model performance with various metrics
model.eval()
with torch.no_grad():
    outputs, joint_repr = model(X1_test, X2_test)
    _, predicted = torch.max(outputs, 1)
    
    # Convert to numpy for sklearn metrics
    y_true = y_test.numpy()
    y_pred = predicted.numpy()
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

# Visualize results
plt.figure(figsize=(15, 10))

# Plot the training and test loss
plt.subplot(2, 2, 1)
plt.plot(range(num_epochs), train_losses, label='Train Loss')
plt.plot(range(num_epochs), test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()

# Plot test accuracy
plt.subplot(2, 2, 2)
plt.plot(range(num_epochs), test_accuracies)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')

# Visualize the joint representation
from sklearn.decomposition import PCA

# Apply PCA to reduce dimensionality for visualization
pca = PCA(n_components=2)
joint_repr_2d = pca.fit_transform(joint_repr.numpy())

# Plot the joint representation colored by class
plt.subplot(2, 2, 3)
for class_idx in range(3):
    mask = (y_test == class_idx).numpy()
    plt.scatter(joint_repr_2d[mask, 0], joint_repr_2d[mask, 1], label=f'Class {class_idx}')
plt.xlabel('PC 1')
plt.ylabel('PC 2')
plt.title('Joint Representation (PCA)')
plt.legend()

# Compare with single-view performance
# Create and train a model using only view 1
class SingleViewNetwork(nn.Module):
    def __init__(self, input_dim=20, hidden_dim=32, num_classes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        return self.network(x)

# Train and evaluate on view 1 only
view1_model = SingleViewNetwork(input_dim=X_view1.shape[1])
view1_optimizer = optim.Adam(view1_model.parameters(), lr=0.001)

# Brief training loop for view 1 model
for epoch in range(num_epochs):
    # Train
    view1_model.train()
    outputs = view1_model(X1_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimize
    view1_optimizer.zero_grad()
    loss.backward()
    view1_optimizer.step()

# Evaluate view 1 model
view1_model.eval()
with torch.no_grad():
    outputs = view1_model(X1_test)
    _, predicted = torch.max(outputs, 1)
    view1_accuracy = (predicted == y_test).float().mean().item()

# Train and evaluate on view 2 only
view2_model = SingleViewNetwork(input_dim=X_view2.shape[1])
view2_optimizer = optim.Adam(view2_model.parameters(), lr=0.001)

# Brief training loop for view 2 model
for epoch in range(num_epochs):
    # Train
    view2_model.train()
    outputs = view2_model(X2_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimize
    view2_optimizer.zero_grad()
    loss.backward()
    view2_optimizer.step()

# Evaluate view 2 model
view2_model.eval()
with torch.no_grad():
    outputs = view2_model(X2_test)
    _, predicted = torch.max(outputs, 1)
    view2_accuracy = (predicted == y_test).float().mean().item()

# Compare accuracies
plt.subplot(2, 2, 4)
accuracies = [view1_accuracy, view2_accuracy, accuracy]
plt.bar(['View 1 Only', 'View 2 Only', 'Multi-View'], accuracies)
plt.ylabel('Accuracy')
plt.title('Performance Comparison')
for i, acc in enumerate(accuracies):
    plt.text(i, acc + 0.01, f'{acc:.4f}', ha='center')

plt.tight_layout()
plt.show()
```


# Example 16: Simple Self-Supervised Learning with Contrastive Loss

```python

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate data for contrastive learning
def generate_clustering_data(num_samples=1000, num_classes=5, dim=50):
    # Generate class centers
    centers = np.random.randn(num_classes, dim) * 5
    
    X = np.zeros((num_samples, dim))
    y = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        # Assign a class
        class_idx = i % num_classes
        y[i] = class_idx
        
        # Generate data point based on class center with noise
        X[i] = centers[class_idx] + np.random.randn(dim) * 1.0
    
    # Shuffle the data
    idx = np.random.permutation(num_samples)
    X, y = X[idx], y[idx]
    
    return X, y

# Generate data
X, y = generate_clustering_data(num_samples=1000, num_classes=5, dim=50)
X_tensor = torch.FloatTensor(X)

# Define the encoder model
class Encoder(nn.Module):
    def __init__(self, input_dim=50, output_dim=20):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

# Define the contrastive loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, features):
        batch_size = features.size(0)
        
        # Create labels for the positive pairs (diagonal elements)
        labels = torch.arange(batch_size).to(features.device)
        
        # Normalize features for cosine similarity
        features_norm = features / features.norm(dim=1, keepdim=True)
        
        # Calculate cosine similarity matrix
        similarity_matrix = torch.matmul(features_norm, features_norm.T) / self.temperature
        
        # Mask out the diagonal (self-similarity)
        mask = torch.eye(batch_size).bool().to(features.device)
        similarity_matrix.masked_fill_(mask, -float('inf'))
        
        # Calculate loss using the positive pairs
        loss = self.criterion(similarity_matrix, labels)
        
        return loss

# Initialize the model and loss
model = Encoder(input_dim=X.shape[1], output_dim=20)
contrastive_loss = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
batch_size = 128
losses = []

for epoch in range(num_epochs):
    # Shuffle the data
    indices = torch.randperm(len(X_tensor))
    X_shuffled = X_tensor[indices]
    
    # Process in batches
    total_loss = 0
    num_batches = 0
    
    for i in range(0, len(X_tensor), batch_size):
        # Get batch
        batch = X_shuffled[i:i + batch_size]
        
        # Forward pass
        features = model(batch)
        loss = contrastive_loss(features)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    # Average loss for the epoch
    avg_loss = total_loss / num_batches
    losses.append(avg_loss)
    
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

# Evaluate the learned representations
model.eval()
with torch.no_grad():
    # Get embeddings for all data points
    embeddings = model(X_tensor).numpy()

# Use t-SNE to visualize the embeddings
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Use a simple classifier on the embeddings to evaluate their quality
# Split data into train and test
train_size = int(0.8 * len(X))
train_indices = np.random.choice(len(X), train_size, replace=False)
test_indices = np.array(list(set(range(len(X))) - set(train_indices)))

embed_train, embed_test = embeddings[train_indices], embeddings[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Train a k-NN classifier on the embeddings
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(embed_train, y_train)
y_pred = knn.predict(embed_test)
accuracy = np.mean(y_pred == y_test)

print(f"k-NN Accuracy on Embeddings: {accuracy:.4f}")

# Visualize results
plt.figure(figsize=(15, 10))

# Plot the training loss
plt.subplot(2, 2, 1)
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Contrastive Learning Loss')

# Visualize the embeddings colored by class
plt.subplot(2, 2, 2)
for class_idx in range(5):
    mask = (y == class_idx)
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'Class {class_idx}')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('Embeddings Visualization (t-SNE)')
plt.legend()

# Compare with t-SNE on raw features
raw_tsne = TSNE(n_components=2, random_state=42).fit_transform(X)
plt.subplot(2, 2, 3)
for class_idx in range(5):
    mask = (y == class_idx)
    plt.scatter(raw_tsne[mask, 0], raw_tsne[mask, 1], label=f'Class {class_idx}')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.title('Raw Data Visualization (t-SNE)')
plt.legend()

# Visualize the similarity matrix
plt.subplot(2, 2, 4)
with torch.no_grad():
    # Take a small subset for visualization
    subset_size = 100
    subset_indices = np.random.choice(len(X), subset_size, replace=False)
    subset_indices = sorted(subset_indices, key=lambda x: y[x])  # Sort by class for better visualization
    
    subset_X = X_tensor[subset_indices]
    subset_y = y[subset_indices]
    
    # Get embeddings and compute similarity
    subset_embeddings = model(subset_X)
    subset_embeddings_norm = subset_embeddings / subset_embeddings.norm(dim=1, keepdim=True)
    similarity = torch.matmul(subset_embeddings_norm, subset_embeddings_norm.T).numpy()

# Plot similarity matrix
plt.imshow(similarity, cmap='viridis')
plt.colorbar(label='Cosine Similarity')
plt.title('Embedding Similarity Matrix')
plt.xlabel('Sample Index')
plt.ylabel('Sample Index')

plt.tight_layout()
plt.show()
```



# Problem 1: Siamese Network for Few-Shot Learning
Problem Description:

Build a Siamese network that can learn to classify images with only a few examples per class. The network should compare new images against a small set of reference images and determine their similarity.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Generate synthetic image data with multiple classes
def generate_class_images(num_classes=10, samples_per_class=20, img_size=28):
    # Create templates for each class (simple patterns)
    templates = []
    for i in range(num_classes):
        template = np.zeros((img_size, img_size))
        
        # Create different patterns for each class
        if i % 3 == 0:  # Circles of different sizes
            center = img_size // 2
            radius = (i % 5 + 2) * img_size // 20
            for x in range(img_size):
                for y in range(img_size):
                    if ((x - center) ** 2 + (y - center) ** 2) < radius ** 2:
                        template[x, y] = 1.0
        
        elif i % 3 == 1:  # Squares of different sizes
            size = (i % 5 + 2) * img_size // 15
            start = (img_size - size) // 2
            template[start:start+size, start:start+size] = 1.0
            
        else:  # Cross patterns of different thicknesses
            thickness = (i % 5 + 1) * 2
            center = img_size // 2
            template[center-thickness//2:center+thickness//2, :] = 1.0
            template[:, center-thickness//2:center+thickness//2] = 1.0
            
        templates.append(template)
    
    # Generate samples for each class with variations
    X = np.zeros((num_classes * samples_per_class, 1, img_size, img_size))
    y = np.zeros(num_classes * samples_per_class, dtype=int)
    
    for i in range(num_classes):
        for j in range(samples_per_class):
            idx = i * samples_per_class + j
            y[idx] = i
            
            # Start with the template
            img = templates[i].copy()
            
            # Add variations
            # 1. Random rotation
            angle = np.random.uniform(-30, 30)
            img = rotate_image(img, angle)
            
            # 2. Random shift
            shift_x = np.random.randint(-3, 4)
            shift_y = np.random.randint(-3, 4)
            img = np.roll(np.roll(img, shift_x, axis=0), shift_y, axis=1)
            
            # 3. Add noise
            img += np.random.normal(0, 0.1, img.shape)
            img = np.clip(img, 0, 1)
            
            X[idx, 0] = img
    
    # Shuffle the data
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    return X, y

def rotate_image(image, angle):
    from scipy.ndimage import rotate
    return rotate(image, angle, reshape=False)

# Generate data
X, y = generate_class_images(num_classes=10, samples_per_class=20)

# Define the Siamese Network
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        
        # CNN Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Linear(128, 64)
        )
        
    def forward_one(self, x):
        return self.encoder(x)
    
    def forward(self, x1, x2):
        # Encode both inputs
        output1 = self.forward_one(x1)
        output2 = self.forward_one(x2)
        
        # Calculate Euclidean distance
        return torch.pairwise_distance(output1, output2)

# Define the contrastive loss function
class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
    
    def forward(self, distance, label):
        # label=0 means similar pair, label=1 means dissimilar pair
        # For similar pairs, we want the distance to be small
        # For dissimilar pairs, we want the distance to be larger than margin
        loss_similar = label * torch.pow(distance, 2)
        loss_dissimilar = (1-label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2)
        return torch.mean(loss_similar + loss_dissimilar)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)

# Create pairs of images for training
def create_pairs(X, y, num_pairs=5000):
    pairs = []
    labels = []
    n_classes = len(np.unique(y))
    
    # For each pair, we store indices of the two images and a label
    # label=0: same class, label=1: different class
    for i in range(num_pairs):
        # Randomly decide if this will be a similar or dissimilar pair
        is_similar = np.random.random() > 0.5
        
        if is_similar:
            # Select a random class
            class_idx = np.random.randint(0, n_classes)
            # Find all indices for this class
            indices = np.where(y == class_idx)[0]
            # Select two random instances from this class
            idx1, idx2 = np.random.choice(indices, 2, replace=False)
            label = 0  # Same class
        else:
            # Select two random classes
            class_idx1, class_idx2 = np.random.choice(n_classes, 2, replace=False)
            # Select one instance from each class
            idx1 = np.random.choice(np.where(y == class_idx1)[0])
            idx2 = np.random.choice(np.where(y == class_idx2)[0])
            label = 1  # Different class
        
        pairs.append([idx1, idx2])
        labels.append(label)
    
    return np.array(pairs), np.array(labels)

# Create training pairs
train_pairs, train_labels = create_pairs(X, y)

# Initialize model, loss, and optimizer
model = SiameseNetwork()
criterion = ContrastiveLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 20
batch_size = 64
losses = []

for epoch in range(num_epochs):
    # Shuffle pairs
    idx = np.random.permutation(len(train_pairs))
    train_pairs_shuffled = train_pairs[idx]
    train_labels_shuffled = train_labels[idx]
    
    # Process in batches
    total_loss = 0
    num_batches = 0
    
    for i in range(0, len(train_pairs_shuffled), batch_size):
        # Get batch
        batch_pairs = train_pairs_shuffled[i:i + batch_size]
        batch_labels = train_labels_shuffled[i:i + batch_size]
        
        # Convert to tensors
        img1 = X_tensor[batch_pairs[:, 0]]
        img2 = X_tensor[batch_pairs[:, 1]]
        labels = torch.FloatTensor(batch_labels)
        
        # Forward pass
        distances = model(img1, img2)
        loss = criterion(distances, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    # Average loss for the epoch
    avg_loss = total_loss / num_batches
    losses.append(avg_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

# Implement few-shot learning
def few_shot_classifier(support_X, support_y, query_X, k=5):
    """
    Support set: examples we've seen (few per class)
    Query set: examples we want to classify
    k: number of nearest neighbors to consider
    """
    model.eval()
    with torch.no_grad():
        # Compute embeddings for support set
        support_embeddings = model.forward_one(support_X)
        
        # Compute embeddings for query set
        query_embeddings = model.forward_one(query_X)
        
        # For each query embedding, find the nearest support embeddings
        predictions = []
        
        for query_emb in query_embeddings:
            # Compute distances to all support embeddings
            distances = torch.pairwise_distance(query_emb.unsqueeze(0), support_embeddings)
            
            # Find k smallest distances
            _, indices = torch.topk(distances, k=k, largest=False)
            
            # Get the classes of the k nearest neighbors
            nearest_classes = support_y[indices]
            
            # Predict the most common class
            prediction = np.bincount(nearest_classes.cpu().numpy()).argmax()
            predictions.append(prediction)
    
    return np.array(predictions)

# Test few-shot learning
# Prepare a support set with few examples per class
n_classes = 10
n_support_per_class = 3  # Few-shot learning with 3 examples per class
n_query_per_class = 10   # Number of test examples per class

# Split data by class for support and query sets
support_indices = []
query_indices = []

for class_idx in range(n_classes):
    # Get all examples for this class
    class_indices = np.where(y == class_idx)[0]
    # Randomly select support and query examples
    support_idx = np.random.choice(class_indices, n_support_per_class, replace=False)
    # Remaining examples go to query set
    remaining_idx = np.setdiff1d(class_indices, support_idx)
    query_idx = np.random.choice(remaining_idx, min(n_query_per_class, len(remaining_idx)), replace=False)
    
    support_indices.extend(support_idx)
    query_indices.extend(query_idx)

# Create support and query sets
support_X = X_tensor[support_indices]
support_y = torch.LongTensor(y[support_indices])
query_X = X_tensor[query_indices]
query_y = y[query_indices]

# Use our few-shot classifier
predictions = few_shot_classifier(support_X, support_y, query_X, k=1)
accuracy = accuracy_score(query_y, predictions)
print(f"Few-shot learning accuracy: {accuracy:.4f}")

# Visualize results
plt.figure(figsize=(15, 10))

# Plot the training loss
plt.subplot(2, 2, 1)
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Siamese Network Training Loss')

# Visualize embedding space using t-SNE
from sklearn.manifold import TSNE

# Get embeddings for a subset of examples
n_vis = 500
vis_indices = np.random.choice(len(X), min(n_vis, len(X)), replace=False)
vis_X = X_tensor[vis_indices]
vis_y = y[vis_indices]

with torch.no_grad():
    embeddings = model.forward_one(vis_X).numpy()

# Apply t-SNE for visualization
tsne = TSNE(n_components=2, random_state=42)
embeddings_2d = tsne.fit_transform(embeddings)

# Plot t-SNE visualization
plt.subplot(2, 2, 2)
for class_idx in range(10):
    mask = (vis_y == class_idx)
    plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'Class {class_idx}')
plt.title('Embedding Space (t-SNE)')
plt.legend()

# Visualize some support and query examples
plt.subplot(2, 2, 3)
plt.title('Support Examples (3 per class)')
for i in range(min(9, len(support_indices))):
    plt.subplot(2, 2, 3 + i//3)
    plt.imshow(X[support_indices[i], 0], cmap='gray')
    plt.title(f'Class {y[support_indices[i]]}')
    plt.axis('off')

plt.subplot(2, 2, 4)
plt.title('Query Examples with Predictions')
for i in range(min(9, len(query_indices))):
    plt.subplot(2, 4, 9 + i)
    plt.imshow(X[query_indices[i], 0], cmap='gray')
    plt.title(f'True: {query_y[i]}, Pred: {predictions[i]}')
    plt.axis('off')

plt.tight_layout()
plt.show()

```

# label to image?
```python

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load MNIST dataset
print("Loading MNIST dataset...")
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype('float32') / 255.0  # Normalize to [0,1]
y = mnist.target.astype('int64')

# Reshape images to (N, 1, 28, 28)
X_reshaped = X.reshape(-1, 1, 28, 28)

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X_reshaped)
y_tensor = torch.LongTensor(y)

# Take a subset for faster training
subset_size = 10000
indices = np.random.choice(len(X_tensor), subset_size, replace=False)
X_subset = X_tensor[indices]
y_subset = y_tensor[indices]

print(f"Using {subset_size} samples from MNIST")

# Define the generator model (from label to image)
class DigitGenerator(nn.Module):
    def __init__(self, num_classes=10, img_size=28):
        super().__init__()
        self.img_size = img_size
        
        # Embedding layer to convert class labels to vectors
        self.embedding = nn.Embedding(num_classes, 64)
        
        # Generator network
        self.generator = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_size * img_size),
            nn.Sigmoid()  # Outputs pixel values in [0, 1]
        )
    
    def forward(self, label):
        # Convert label to embedding
        embedded = self.embedding(label)
        # Generate image
        img = self.generator(embedded)
        # Reshape to image dimensions
        img = img.view(-1, 1, self.img_size, self.img_size)
        return img

# Initialize the model
model = DigitGenerator()

# Define loss function and optimizer
criterion = nn.MSELoss()  # Mean Squared Error for image reconstruction
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 50
batch_size = 64

# Split data into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")

train_losses = []
val_losses = []

for epoch in range(num_epochs):
    # Training phase
    model.train()
    
    # Shuffle the training data
    indices = torch.randperm(len(X_train))
    X_shuffled = X_train[indices]
    y_shuffled = y_train[indices]
    
    total_train_loss = 0
    num_train_batches = 0
    
    for i in range(0, len(X_train), batch_size):
        # Get batch
        batch_X = X_shuffled[i:i + batch_size]
        batch_y = y_shuffled[i:i + batch_size]
        
        # Generate images from labels
        generated_images = model(batch_y)
        
        # Calculate loss
        loss = criterion(generated_images, batch_X)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        num_train_batches += 1
    
    avg_train_loss = total_train_loss / num_train_batches
    train_losses.append(avg_train_loss)
    
    # Validation phase
    model.eval()
    with torch.no_grad():
        total_val_loss = 0
        num_val_batches = 0
        
        for i in range(0, len(X_val), batch_size):
            # Get batch
            batch_X = X_val[i:i + batch_size]
            batch_y = y_val[i:i + batch_size]
            
            # Generate images from labels
            generated_images = model(batch_y)
            
            # Calculate loss
            loss = criterion(generated_images, batch_X)
            
            total_val_loss += loss.item()
            num_val_batches += 1
        
        avg_val_loss = total_val_loss / num_val_batches
        val_losses.append(avg_val_loss)
    
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

# Plot the training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(range(1, num_epochs+1), train_losses, label='Training Loss')
plt.plot(range(1, num_epochs+1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()

# Function to generate digit images
def generate_digit_image(digit, model):
    """Generate an image for a given digit"""
    model.eval()
    with torch.no_grad():
        # Convert digit to tensor
        digit_tensor = torch.LongTensor([digit])
        # Generate image
        image = model(digit_tensor)
        # Convert to numpy
        image_np = image.squeeze().numpy()
    return image_np

# Show examples of real and generated digits
plt.figure(figsize=(15, 6))

# Original MNIST examples
plt.subplot(2, 1, 1)
plt.title("Original MNIST Examples")
for i in range(10):
    # Find an example of each digit
    digit_idx = np.where(y == i)[0][0]
    plt.subplot(2, 10, i+1)
    plt.imshow(X_reshaped[digit_idx, 0], cmap='gray')
    plt.title(f"Digit {i}")
    plt.axis('off')

# Generated images
plt.subplot(2, 1, 2)
plt.title("Generated Digit Images")
for i in range(10):
    plt.subplot(2, 10, i+11)
    gen_image = generate_digit_image(i, model)
    plt.imshow(gen_image, cmap='gray')
    plt.title(f"Generated {i}")
    plt.axis('off')

plt.tight_layout()
plt.show()

# Simple interactive function
def display_digit(digit):
    """Display the generated image for a given digit"""
    if digit < 0 or digit > 9:
        print("Please enter a digit between 0 and 9")
        return
    
    # Find a real example of the digit
    digit_idx = np.where(y == digit)[0][0]
    real_image = X_reshaped[digit_idx, 0]
    
    # Generate an image for the digit
    gen_image = generate_digit_image(digit, model)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(real_image, cmap='gray')
    plt.title(f"Real MNIST Digit {digit}")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(gen_image, cmap='gray')
    plt.title(f"Generated Digit {digit}")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

# Example usage
print("Displaying real and generated images for digits 0 and 1:")
display_digit(0)
display_digit(1)
```
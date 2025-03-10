# PyTorch Deep Learning Tutorial: From Scratch to Advanced

*Current as of 2025-03-04*

## Table of Contents
1. [Introduction to PyTorch](#1-introduction-to-pytorch)
2. [Tensors: The Foundation](#2-tensors-the-foundation)
3. [Autograd: Automatic Differentiation](#3-autograd-automatic-differentiation)
4. [Building Neural Networks](#4-building-neural-networks)
5. [Training Loop Basics](#5-training-loop-basics)
6. [Binary Classification](#6-binary-classification)
7. [Multi-class Classification](#7-multi-class-classification)
8. [Regression Problems](#8-regression-problems)
9. [Optimization Techniques](#9-optimization-techniques)
10. [Advanced PyTorch Features](#10-advanced-pytorch-features)

## 1. Introduction to PyTorch

PyTorch is an open-source machine learning library developed by Facebook's AI Research lab. It's known for its flexibility and dynamic computational graph, which makes debugging and experimentation easier.

### Installation

```bash
# Installing PyTorch with CUDA support
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117

# For CPU-only version
pip install torch torchvision torchaudio
```

To check if PyTorch is installed correctly and if CUDA is available:

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
```

## 2. Tensors: The Foundation

Tensors are the fundamental data structure in PyTorch, similar to NumPy arrays but with GPU acceleration capabilities.

```python
import torch
import numpy as np

# Creating tensors
x = torch.tensor([1, 2, 3, 4])
print(f"Tensor: {x}")

# From NumPy
np_array = np.array([1, 2, 3, 4])
x_np = torch.from_numpy(np_array)
print(f"From NumPy: {x_np}")

# Various initialization methods
zeros = torch.zeros(3, 4)  # 3x4 tensor of zeros
ones = torch.ones(2, 3)    # 2x3 tensor of ones
rand = torch.rand(2, 4)    # 2x4 tensor of random values from uniform distribution [0,1)
randn = torch.randn(2, 4)  # 2x4 tensor of random values from normal distribution (mean=0, std=1)

print(f"Zeros:\n{zeros}")
print(f"Ones:\n{ones}")
print(f"Random uniform:\n{rand}")
print(f"Random normal:\n{randn}")

# Tensor properties
print(f"Shape: {rand.shape}")
print(f"Data type: {rand.dtype}")
print(f"Device: {rand.device}")  # CPU or GPU

# Basic operations
a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

print(f"a + b: {a + b}")
print(f"a * b: {a * b}")  # Element-wise multiplication
print(f"Matrix multiplication: {torch.matmul(a, b)}")

# Reshaping
c = torch.randn(4, 4)
print(f"Original:\n{c}")
print(f"Reshaped to 2x8:\n{c.view(2, 8)}")
print(f"Transposed:\n{c.t()}")

# Moving to GPU (if available)
if torch.cuda.is_available():
    x_gpu = x.to("cuda")
    print(f"Device after moving to GPU: {x_gpu.device}")
```

## 3. Autograd: Automatic Differentiation

PyTorch's autograd system enables automatic computation of gradients, essential for training neural networks.

```python
import torch

# Create tensors with gradient tracking
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Perform operations
z = x**2 + y**3

# Compute gradients
z.backward()

# Access gradients
print(f"dz/dx: {x.grad}")  # Should be 2*x = 4
print(f"dz/dy: {y.grad}")  # Should be 3*y^2 = 27

# More complex example with vector operations
weights = torch.randn(4, 3, requires_grad=True)
inputs = torch.randn(2, 4)

# Forward pass
outputs = torch.matmul(inputs, weights)
loss = outputs.sum()

# Backward pass
loss.backward()

print(f"Weights gradient shape: {weights.grad.shape}")
print(f"Weights gradient:\n{weights.grad}")

# Detaching from computation graph
a = torch.randn(2, 2, requires_grad=True)
b = a * 2
c = b.detach()  # Detaches from the computation graph
d = c * 3
e = a * 5

# This works because a has requires_grad=True
e.sum().backward()
print(f"Gradient of a: {a.grad}")  # Should be 5.0

# To avoid accumulating gradients
a.grad.zero_()
```

## 4. Building Neural Networks

PyTorch provides the `nn` module to build neural networks easily.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model
input_size = 10
hidden_size = 20
output_size = 5
model = SimpleNN(input_size, hidden_size, output_size)

print(model)

# Generate random input
x = torch.randn(3, input_size)  # Batch of 3 samples

# Forward pass
output = model(x)
print(f"Output shape: {output.shape}")  # Should be [3, 5]

# Common activation functions
activation_input = torch.randn(5)
print(f"Input: {activation_input}")
print(f"ReLU: {F.relu(activation_input)}")
print(f"Sigmoid: {torch.sigmoid(activation_input)}")
print(f"Tanh: {torch.tanh(activation_input)}")
print(f"Softmax: {F.softmax(activation_input, dim=0)}")

# Sequential model (alternative way to define networks)
seq_model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 15),
    nn.ReLU(),
    nn.Linear(15, 5)
)

print("\nSequential model:")
print(seq_model)
```

## 5. Training Loop Basics

The basic training loop in PyTorch involves forward pass, loss computation, backward pass, and parameter updates.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# Generate synthetic data
X = torch.randn(1000, 10)
true_weights = torch.randn(10, 1)
y = X @ true_weights + 0.1 * torch.randn(1000, 1)  # Linear relation with some noise

# Create Dataset and DataLoader
dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define model
model = nn.Linear(10, 1)

# Loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward pass and optimize
        optimizer.zero_grad()  # Zero the gradient buffers
        loss.backward()        # Compute gradients
        optimizer.step()       # Update weights
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

# Plot the loss curve
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title("Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()

# Compare learned weights with true weights
print("Learned vs True Weights:")
print(f"Learned: {model.weight.data.flatten()}")
print(f"True: {true_weights.flatten()}")
```

## 6. Binary Classification

Let's implement a binary classification model to predict whether a sample belongs to a certain class.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Generate a non-linear dataset
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
X = torch.FloatTensor(X)
y = torch.FloatTensor(y).unsqueeze(1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create dataloaders
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Define the model for binary classification
class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(2, 16)
        self.layer2 = nn.Linear(16, 16)
        self.layer3 = nn.Linear(16, 1)
    
    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        x = torch.sigmoid(self.layer3(x))  # Sigmoid for binary classification
        return x

# Initialize model, loss, and optimizer
model = BinaryClassifier()
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    train_epoch_loss = 0.0
    
    for batch_X, batch_y in train_loader:
        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_epoch_loss += loss.item()
    
    train_losses.append(train_epoch_loss / len(train_loader))
    
    # Evaluation
    model.eval()
    test_epoch_loss = 0.0
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            test_epoch_loss += loss.item()
    
    test_losses.append(test_epoch_loss / len(test_loader))
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_losses[-1]:.4f}, "
              f"Test Loss: {test_losses[-1]:.4f}")

# Plot training and test loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Evaluate the model's performance
model.eval()
with torch.no_grad():
    y_pred = model(X_test)
    y_pred_class = (y_pred > 0.5).float()

# Convert tensors to numpy for sklearn metrics
y_test_np = y_test.numpy()
y_pred_np = y_pred_class.numpy()

# Calculate metrics
accuracy = accuracy_score(y_test_np, y_pred_np)
precision = precision_score(y_test_np, y_pred_np)
recall = recall_score(y_test_np, y_pred_np)
f1 = f1_score(y_test_np, y_pred_np)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# Visualize decision boundary
def plot_decision_boundary(X, y, model):
    # Set min and max values
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Create a mesh grid
    xx, yy = torch.meshgrid(torch.linspace(x_min, x_max, 100),
                           torch.linspace(y_min, y_max, 100))
    
    # Get predictions
    Z = model(torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1)], dim=1))
    Z = Z.reshape(xx.shape)
    
    # Plot decision boundary and points
    plt.figure(figsize=(10, 8))
    plt.contourf(xx.numpy(), yy.numpy(), Z.numpy(), alpha=0.8, cmap='RdBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap='RdBu')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.colorbar()
    plt.show()

plot_decision_boundary(X, y, model)
```

## 7. Multi-class Classification

Now let's tackle multi-class classification using the MNIST dataset.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns

# Load MNIST dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000)

# Display some images
def show_images(dataloader):
    batch_X, batch_y = next(iter(dataloader))
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        img = batch_X[i].squeeze().numpy()
        ax.imshow(img, cmap='gray')
        ax.set_title(f"Label: {batch_y[i].item()}")
        ax.axis('off')
    plt.tight_layout()
    plt.show()

show_images(train_loader)

# Define the model for multi-class classification
class MNISTClassifier(nn.Module):
    def __init__(self):
        super(MNISTClassifier, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)  # 10 classes for digits 0-9
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)  # No activation here (will use CrossEntropyLoss)
        return x

# Initialize model, loss, and optimizer
model = MNISTClassifier()
criterion = nn.CrossEntropyLoss()  # Combines LogSoftmax and NLLLoss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Function to calculate accuracy
def calculate_accuracy(model, dataloader, device='cpu'):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

# Training loop
num_epochs = 10
train_losses = []
train_accuracies = []
test_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    
    for i, (inputs, labels) in enumerate(train_loader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    # Calculate metrics at the end of each epoch
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    train_accuracy = calculate_accuracy(model, train_loader)
    train_accuracies.append(train_accuracy)
    
    test_accuracy = calculate_accuracy(model, test_loader)
    test_accuracies.append(test_accuracy)
    
    print(f"Epoch [{epoch+1}/{num_epochs}], "
          f"Train Loss: {train_loss:.4f}, "
          f"Train Accuracy: {train_accuracy:.2f}%, "
          f"Test Accuracy: {test_accuracy:.2f}%")

# Plot training metrics
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train')
plt.plot(test_accuracies, label='Test')
plt.title('Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.show()

# Confusion matrix
def plot_confusion_matrix(model, dataloader, device='cpu'):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

plot_confusion_matrix(model, test_loader)

# Visualize misclassified examples
def show_misclassified(model, dataloader, num_images=10, device='cpu'):
    model.eval()
    misclassified_images = []
    misclassified_labels = []
    misclassified_preds = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            # Find misclassified indices
            misclassified_idx = (preds != labels).nonzero(as_tuple=True)[0]
            
            for idx in misclassified_idx:
                if len(misclassified_images) < num_images:
                    misclassified_images.append(inputs[idx].cpu())
                    misclassified_labels.append(labels[idx].item())
                    misclassified_preds.append(preds[idx].item())
                else:
                    break
            
            if len(misclassified_images) >= num_images:
                break
    
    # Display misclassified images
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        if i < len(misclassified_images):
            img = misclassified_images[i].squeeze().numpy()
            ax.imshow(img, cmap='gray')
            ax.set_title(f"True: {misclassified_labels[i]}\nPred: {misclassified_preds[i]}")
            ax.axis('off')
    
    plt.tight_layout()
    plt.show()

show_misclassified(model, test_loader)

# Save the model
torch.save(model.state_dict(), 'mnist_classifier.pth')
print("Model saved to 'mnist_classifier.pth'")
```

## 8. Regression Problems

Now let's implement a regression model.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load California housing dataset
housing = fetch_california_housing()
X, y = housing.data, housing.target

# Standardize features and target
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test)

# Create dataloaders
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64)

# Define the regression model
class HousingRegressor(nn.Module):
    def __init__(self, input_dim):
        super(HousingRegressor, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.layer1(x)))
        x = self.dropout(x)
        x = self.relu(self.bn2(self.layer2(x)))
        x = self.dropout(x)
        x = self.layer3(x)
        return x.squeeze()

# Initialize model, loss, and optimizer
input_dim = X_train.shape[1]
model = HousingRegressor(input_dim)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # L2 regularization

# Function to calculate metrics
def calculate_metrics(model, dataloader):
    model.eval()
    total_mse = 0
    predictions = []
    actual_values = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_mse += loss.item()
            
            predictions.extend(outputs.cpu().numpy())
            actual_values.extend(targets.cpu().numpy())
    
    # Calculate metrics
    mse = total_mse / len(dataloader)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(np.array(predictions) - np.array(actual_values)))
    r2 = 1 - np.sum((np.array(actual_values) - np.array(predictions))**2) / np.sum((np.array(actual_values) - np.mean(actual_values))**2)
    
    return mse, rmse, mae, r2, predictions, actual_values

# Training loop
num_epochs = 100
train_losses = []
test_losses = []

for epoch in range(num_epochs):
    # Training
    model.train()
    running_loss = 0.0
    
    for inputs, targets in train_loader:
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    # Calculate losses
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)
    
    # Evaluate on test set
    model.eval()
    test_running_loss = 0.0
    with torch.no_grad():
        for inputs, targets in test_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_running_loss += loss.item()
    
    test_loss = test_running_loss / len(test_loader)
    test_losses.append(test_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {train_loss:.4f}, "
              f"Test Loss: {test_loss:.4f}")

# Final evaluation
mse, rmse, mae, r2, predictions, actual_values = calculate_metrics(model, test_loader)
print(f"Test MSE: {mse:.4f}")
print(f"Test RMSE: {rmse:.4f}")
print(f"Test MAE: {mae:.4f}")
print(f"Test R²: {r2:.4f}")

# Plot training and test loss
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('MSE Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.show()

# Plot predictions vs actual values
plt.figure(figsize=(10, 6))
plt.scatter(actual_values, predictions, alpha=0.5)
plt.plot([min(actual_values), max(actual_values)], [min(actual_values), max(actual_values)], 'r--')
plt.xlabel('Actual Values (Standardized)')
plt.ylabel('Predictions (Standardized)')
plt.title('Predictions vs Actual Values')
plt.show()

# Function to transform predictions back to original scale
def inverse_transform_predictions(preds, actuals):
    preds_original = scaler_y.inverse_transform(preds.reshape(-1, 1)).flatten()
    actuals_original = scaler_y.inverse_transform(actuals.reshape(-1, 1)).flatten()
    return preds_original, actuals_original

# Convert predictions back to original scale
preds_orig, actuals_orig = inverse_transform_predictions(np.array(predictions), np.array(actual_values))

# Calculate metrics in original scale
mse_orig = np.mean((preds_orig - actuals_orig) ** 2)
rmse_orig = np.sqrt(mse_orig)
mae_orig = np.mean(np.abs(preds_orig - actuals_orig))

print(f"Original scale - MSE: {mse_orig:.4f}")
print(f"Original scale - RMSE: {rmse_orig:.4f}")
print(f"Original scale - MAE: {mae_orig:.4f}")

# Plot predictions vs actual values in original scale
plt.figure(figsize=(10, 6))
plt.scatter(actuals_orig, preds_orig, alpha=0.5)
plt.plot([min(actuals_orig), max(actuals_orig)], [min(actuals_orig), max(actuals_orig)], 'r--')
plt.xlabel('Actual Values')
plt.ylabel('Predictions')
plt.title('Predictions vs Actual Values (Original Scale)')
plt.show()

# Save the model
torch.save(model.state_dict(), 'housing_regressor.pth')
print("Model saved to 'housing_regressor.pth'")
```

## 9. Optimization Techniques

Optimizing neural networks involves more than just using basic optimizers. Let's explore various techniques.

### 9.1 Different Optimizers

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 20).astype(np.float32)
w_true = np.random.randn(20, 1).astype(np.float32)
y = X @ w_true + 0.1 * np.random.randn(1000, 1).astype(np.float32)

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)

# Create dataset and dataloader
dataset = TensorDataset(X_tensor, y_tensor)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Define a simple model
class LinearModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearModel, self).__init__()
        self.fc = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.fc(x)

# Function to train model with different optimizers
def train_with_optimizer(optimizer_name, dataloader, lr=0.01, weight_decay=0):
    # Initialize model
    model = LinearModel(20)
    
    # Reset parameters to same initialization
    torch.manual_seed(42)
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    # Choose optimizer
    if optimizer_name == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Momentum':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    elif optimizer_name == 'Adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'RMSprop':
        optimizer = optim.RMSprop(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Loss function
    criterion = nn.MSELoss()
    
    # Training loop
    losses = []
    for epoch in range(100):
        epoch_loss = 0.0
        for X_batch, y_batch in dataloader:
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
    
    return losses

# Compare different optimizers
optimizers = ['SGD', 'Momentum', 'Adagrad', 'RMSprop', 'Adam']
all_losses = {}

for opt in optimizers:
    print(f"Training with {opt}...")
    losses = train_with_optimizer(opt, dataloader)
    all_losses[opt] = losses

# Plot the loss curves
plt.figure(figsize=(12, 6))
for opt, losses in all_losses.items():
    plt.plot(losses, label=opt)

plt.title('Optimizer Comparison')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.legend()
plt.grid(True)
plt.show()
```

### 9.2 Learning Rate Schedulers

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Define a simple model
model = nn.Sequential(
    nn.Linear(10, 50),
    nn.ReLU(),
    nn.Linear(50, 1)
)

# Base optimizer
optimizer = optim.Adam(model.parameters(), lr=0.1)

# Define different schedulers
schedulers = {
    'StepLR': StepLR(optimizer, step_size=20, gamma=0.5),
    'ExponentialLR': ExponentialLR(optimizer, gamma=0.975),
    'CosineAnnealingLR': CosineAnnealingLR(optimizer, T_max=100),
    'ReduceLROnPlateau': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
}

# Simulate learning rates over epochs
epochs = 100
learning_rates = {name: [] for name in schedulers.keys()}

# Generate learning rate curves
for name, scheduler in schedulers.items():
    # Reset optimizer's learning rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = 0.1
    
    # Record learning rates
    for epoch in range(epochs):
        if name == 'ReduceLROnPlateau':
            # ReduceLROnPlateau needs loss value
            val_loss = 1.0 / (epoch + 1)  # Simulated decreasing loss
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        learning_rates[name].append(current_lr)

# Plot learning rate schedules
plt.figure(figsize=(12, 6))
for name, lrs in learning_rates.items():
    plt.plot(lrs, label=name)

plt.title('Learning Rate Schedules')
plt.xlabel('Epochs')
plt.ylabel('Learning Rate')
plt.legend()
plt.grid(True)
plt.show()
```

### 9.3 Weight Initialization Techniques

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# Define a deep network to test initialization methods
class DeepNet(nn.Module):
    def __init__(self, depth=10, width=100, init_method='default'):
        super(DeepNet, self).__init__()
        self.layers = nn.ModuleList()
        
        # Input layer
        self.layers.append(nn.Linear(10, width))
        
        # Hidden layers
        for _ in range(depth - 2):
            self.layers.append(nn.Linear(width, width))
        
        # Output layer
        self.layers.append(nn.Linear(width, 1))
        
        # Initialize weights
        self.apply_init(init_method)
    
    def apply_init(self, method):
        for layer in self.layers:
            if method == 'default':
                # PyTorch default (uniform)
                pass
            elif method == 'xavier_uniform':
                nn.init.xavier_uniform_(layer.weight)
            elif method == 'xavier_normal':
                nn.init.xavier_normal_(layer.weight)
            elif method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            elif method == 'kaiming_normal':
                nn.init.kaiming_normal_(layer.weight, nonlinearity='relu')
            elif method == 'orthogonal':
                nn.init.orthogonal_(layer.weight)
            
            # Initialize biases to zero
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # ReLU for all but the last layer
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x

# Analyze activation distributions with different initializations
def analyze_activations(init_method):
    torch.manual_seed(42)
    model = DeepNet(init_method=init_method)
    model.eval()
    
    # Generate random input
    x = torch.randn(1000, 10)
    
    # Collect activations from each layer
    activations = []
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            x = layer(x)
            if i < len(model.layers) - 1:
                x = torch.relu(x)
                # Sample some activations for visualization
                sample_activations = x[:, :50].flatten().numpy()
                activations.append(sample_activations)
    
    return activations

# Compare different initialization methods
init_methods = ['default', 'xavier_uniform', 'xavier_normal', 
                'kaiming_uniform', 'kaiming_normal', 'orthogonal']

plt.figure(figsize=(15, 10))
for i, method in enumerate(init_methods):
    activations = analyze_activations(method)
    
    for j, layer_activations in enumerate(activations[:3]):  # Show first 3 layers
        plt.subplot(len(init_methods), 3, i*3 + j + 1)
        plt.hist(layer_activations, bins=50)
        plt.title(f"{method} - Layer {j+1}")
        plt.xlim(-1.5, 1.5)
        if j == 0:
            plt.ylabel(method)

plt.tight_layout()
plt.show()
```

### 9.4 Regularization Techniques

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data that is prone to overfitting
np.random.seed(42)
n_samples = 50
X = np.linspace(0, 1, n_samples).reshape(-1, 1).astype(np.float32)
y = 0.5 * np.sin(6 * X) + 0.5 * X + 0.1 * np.random.randn(n_samples, 1).astype(np.float32)

# Create train/test split
train_idx = np.random.choice(n_samples, int(0.7 * n_samples), replace=False)
test_idx = np.array(list(set(range(n_samples)) - set(train_idx)))

X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

X_train_tensor = torch.from_numpy(X_train)
y_train_tensor = torch.from_numpy(y_train)
X_test_tensor = torch.from_numpy(X_test)
y_test_tensor = torch.from_numpy(y_test)

# Define models with different regularization approaches
class NonRegularizedModel(nn.Module):
    def __init__(self):
        super(NonRegularizedModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class L2RegularizedModel(nn.Module):
    def __init__(self):
        super(L2RegularizedModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 1)
        )
    
    def forward(self, x):
        return self.net(x)

class DropoutRegularizedModel(nn.Module):
    def __init__(self):
        super(DropoutRegularizedModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(100, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# Train function with regularization options
def train_model(model, X_train, y_train, X_test, y_test, weight_decay=0):
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    
    train_losses = []
    test_losses = []
    
    for epoch in range(500):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            train_preds = model(X_train)
            train_loss = criterion(train_preds, y_train).item()
            train_losses.append(train_loss)
            
            test_preds = model(X_test)
            test_loss = criterion(test_preds, y_test).item()
            test_losses.append(test_loss)
    
    # Get predictions for plotting
    model.eval()
    with torch.no_grad():
        X_full = torch.from_numpy(np.linspace(0, 1, 1000).reshape(-1, 1).astype(np.float32))
        preds = model(X_full)
    
    return train_losses, test_losses, X_full.numpy(), preds.numpy()

# Train different models
torch.manual_seed(42)
non_reg_model = NonRegularizedModel()
non_reg_train, non_reg_test, X_plot, non_reg_preds = train_model(
    non_reg_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
)

torch.manual_seed(42)
l2_model = L2RegularizedModel()
l2_train, l2_test, _, l2_preds = train_model(
    l2_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, weight_decay=0.01
)

torch.manual_seed(42)
dropout_model = DropoutRegularizedModel()
dropout_train, dropout_test, _, dropout_preds = train_model(
    dropout_model, X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor
)

# Plot results
plt.figure(figsize=(15, 10))

# Plot the loss curves
plt.subplot(2, 1, 1)
plt.plot(non_reg_train, label='No Regularization - Train')
plt.plot(non_reg_test, label='No Regularization - Test')
plt.plot(l2_train, label='L2 Regularization - Train')
plt.plot(l2_test, label='L2 Regularization - Test')
plt.plot(dropout_train, label='Dropout - Train')
plt.plot(dropout_test, label='Dropout - Test')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True)

# Plot the predictions
plt.subplot(2, 1, 2)
plt.scatter(X_train, y_train, label='Training Data', alpha=0.6)
plt.scatter(X_test, y_test, label='Test Data', alpha=0.6)
plt.plot(X_plot, non_reg_preds, label='No Regularization', color='red')
plt.plot(X_plot, l2_preds, label='L2 Regularization', color='blue')
plt.plot(X_plot, dropout_preds, label='Dropout', color='green')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Model Predictions')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
```

## 10. Advanced PyTorch Features

### 10.1 Custom Datasets and DataLoaders

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt

# Example 1: Custom Dataset for CSV data
class CSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform
    
    def __len__(self):
        return len(self.data_frame)
    
    def __getitem__(self, idx):
        # Get features and target from dataframe
        features = torch.tensor(self.data_frame.iloc[idx, :-1].values, dtype=torch.float32)
        target = torch.tensor(self.data_frame.iloc[idx, -1], dtype=torch.float32)
        
        # Apply transforms if any
        if self.transform:
            features = self.transform(features)
        
        return features, target

# Example 2: Custom Dataset for image data
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on an image.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, fname) 
                           for fname in os.listdir(root_dir) 
                           if fname.endswith(('.png', '.jpg', '.jpeg'))]
        
        # For demonstration, we'll create dummy labels
        self.labels = torch.randint(0, 10, (len(self.image_paths),))
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

# Example usage of custom datasets
# Assuming you have a CSV file and image directory
# csv_dataset = CSVDataset('data.csv')
# image_dataset = ImageDataset('images_dir', transform=transforms.ToTensor())

# Creating a DataLoader with custom collate function
def custom_collate(batch):
    """
    Custom collate function to handle batches with different sizes
    or special processing requirements
    """
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    
    # Example: Padding sequences to equal length
    # This is simplified; in practice, you'd use something like:
    # data = pad_sequence(data, batch_first=True, padding_value=0)
    
    return torch.stack(data), torch.tensor(target)

# Example usage of custom collate function
# dataloader = DataLoader(dataset, batch_size=16, collate_fn=custom_collate)

# Creating data with variable lengths (for demonstration)
class VariableLengthDataset(Dataset):
    def __init__(self, num_items=100):
        self.data = [torch.randn(np.random.randint(1, 10)) for _ in range(num_items)]
        self.labels = torch.randint(0, 2, (num_items,))
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# Better collate function that handles variable length sequences
def variable_length_collate(batch):
    # Sort batch by sequence length (descending)
    batch.sort(key=lambda x: len(x[0]), reverse=True)
    
    # Get lengths, sequences, and labels
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    lengths = [len(seq) for seq in sequences]
    
    # Pad sequences to the length of the longest one
    max_length = max(lengths)
    padded_seqs = torch.zeros(len(batch), max_length)
    
    for i, seq in enumerate(sequences):
        end = lengths[i]
        padded_seqs[i, :end] = seq
    
    # Return padded sequences, original lengths (useful for RNNs), and labels
    return padded_seqs, torch.tensor(lengths), torch.tensor(labels)

# Demonstrate using the variable length dataset
var_dataset = VariableLengthDataset()
var_dataloader = DataLoader(var_dataset, batch_size=4, collate_fn=variable_length_collate)

# Get a batch from the dataloader
padded_seqs, lengths, labels = next(iter(var_dataloader))
print(f"Padded sequences shape: {padded_seqs.shape}")
print(f"Lengths: {lengths}")
print(f"Labels: {labels}")
```

### 10.2 Custom Layers and Modules

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 1. Custom layer with learnable parameters
class CustomLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(CustomLinear, self).__init__()
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.randn(out_features))
        else:
            self.register_parameter('bias', None)
        
        # Initialize weights
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # Matrix multiplication + bias if present
        output = torch.matmul(x, self.weight.t())
        if self.bias is not None:
            output += self.bias
        return output

# 2. Custom activation function
class CustomReLU(nn.Module):
    def __init__(self, alpha=0.1):
        super(CustomReLU, self).__init__()
        self.alpha = alpha
        
    def forward(self, x):
        return torch.where(x > 0, x, self.alpha * x)  # Leaky ReLU with custom alpha

# 3. Custom residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Skip connection
        out = F.relu(out)
        return out

# 4. Custom attention mechanism
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        
        # Query, Key, Value projections
        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)
        
        # Output projection
        self.fc_out = nn.Linear(embed_size, embed_size)
        
        # Scaling factor for dot product attention
        self.scale = torch.sqrt(torch.FloatTensor([embed_size]))
    
    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        
        # Get Query, Key, Value projections
        Q = self.query(x)  # [batch_size, seq_len, embed_size]
        K = self.key(x)    # [batch_size, seq_len, embed_size]
        V = self.value(x)  # [batch_size, seq_len, embed_size]
        
        # Compute attention scores
        # Q @ K.transpose(-2, -1) gives [batch_size, seq_len, seq_len]
        attention = torch.matmul(Q, K.transpose(-2, -1)) / self.scale.to(x.device)
        
        # Apply softmax to get attention weights
        attention = F.softmax(attention, dim=-1)
        
        # Apply attention weights to value
        x = torch.matmul(attention, V)  # [batch_size, seq_len, embed_size]
        
        # Output projection
        x = self.fc_out(x)
        
        return x

# Usage examples
if __name__ == "__main__":
    # Test custom linear layer
    custom_linear = CustomLinear(10, 5)
    x = torch.randn(3, 10)
    output = custom_linear(x)
    print(f"Custom linear output shape: {output.shape}")
    
    # Test custom activation
    custom_relu = CustomReLU(alpha=0.2)
    x = torch.randn(3, 5)
    output = custom_relu(x)
    print(f"Custom ReLU min value: {output.min().item()}")
    
    # Test residual block
    res_block = ResidualBlock(64, 128, stride=2)
    x = torch.randn(2, 64, 32, 32)
    output = res_block(x)
    print(f"Residual block output shape: {output.shape}")
    
    # Test self-attention
    attention = SelfAttention(128)
    x = torch.randn(2, 10, 128)  # [batch_size, seq_len, embed_dim]
    output = attention(x)
    print(f"Self-attention output shape: {output.shape}")
```

### 10.3 Distributed Training

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    """Initialize the distributed environment."""
    # Windows: 'gloo' | Linux: 'nccl' if GPU available else 'gloo'
    backend = 'gloo'  # Change to 'nccl' for GPU training on Linux
    
    # Initialize process group
    dist.init_process_group(backend=backend, 
                           init_method='tcp://localhost:12345', 
                           world_size=world_size, 
                           rank=rank)
    
    # Set device for this process
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the distributed environment."""
    dist.destroy_process_group()

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)
    
    def forward(self, x):
        return self.fc(x)

def train(rank, world_size):
    # Initialize the distributed environment
    setup(rank, world_size)
    
    # Create model and move it to the correct device
    model = SimpleModel().to(rank)
    
    # Wrap model with DDP
    ddp_model = DDP(model, device_ids=[rank])
    
    # Create dummy dataset
    X = torch.randn(1000, 10)
    y = torch.randn(1000, 1)
    dataset = TensorDataset(X, y)
    
    # Use DistributedSampler to partition data among workers
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = DataLoader(dataset, batch_size=32, sampler=sampler)
    
    # Create optimizer and loss function
    optimizer = optim.Adam(ddp_model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(5):
        # Set epoch for the sampler
        sampler.set_epoch(epoch)
        
        for batch_idx, (data, target) in enumerate(dataloader):
            # Move data to the correct device
            data, target = data.to(rank), target.to(rank)
            
            # Forward pass
            output = ddp_model(data)
            loss = criterion(output, target)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if batch_idx % 10 == 0:
                print(f"Rank {rank} | Epoch {epoch} | Batch {batch_idx} | Loss {loss.item()}")
    
    # Clean up
    cleanup()

def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)

# Example usage:
# if __name__ == "__main__":
#     world_size = torch.cuda.device_count()
#     run_demo(train, world_size)
```

### 10.4 Custom Loss Functions

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 1. Focal Loss for imbalanced classification
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability 0
        focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return focal_loss.mean()

# 2. Dice Loss for segmentation tasks
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, inputs, targets):
        # Flatten inputs and targets
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (inputs * targets).sum()
        dice = (2. * intersection + self.smooth) / (inputs.sum() + targets.sum() + self.smooth)
        
        return 1 - dice

# 3. Triplet Loss for embedding learning
class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        
    def forward(self, anchor, positive, negative):
        pos_dist = torch.sum((anchor - positive) ** 2, dim=1)
        neg_dist = torch.sum((anchor - negative) ** 2, dim=1)
        loss = F.relu(pos_dist - neg_dist + self.margin)
        return loss.mean()

# 4. Custom L1 smooth loss (Huber Loss)
class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta
        
    def forward(self, inputs, targets):
        diff = torch.abs(inputs - targets)
        cond = diff < self.beta
        loss = torch.where(cond, 0.5 * diff**2 / self.beta, diff - 0.5 * self.beta)
        return loss.mean()

# Visualize the behavior of different loss functions
def visualize_losses():
    # Range of predictions for binary classification
    y_pred = torch.linspace(0, 1, 100)
    
    # Different target values
    y_true_0 = torch.zeros_like(y_pred)  # Target is 0
    y_true_1 = torch.ones_like(y_pred)   # Target is 1
    
    # Standard losses
    bce_loss_0 = F.binary_cross_entropy(y_pred, y_true_0, reduction='none')
    bce_loss_1 = F.binary_cross_entropy(y_pred, y_true_1, reduction='none')
    
    # Custom losses
    focal_loss = FocalLoss(gamma=2.0)
    focal_0 = torch.zeros_like(y_pred)
    focal_1 = torch.zeros_like(y_pred)
    
    for i, p in enumerate(y_pred):
        focal_0[i] = focal_loss(p.view(1), y_true_0[i].view(1))
        focal_1[i] = focal_loss(p.view(1), y_true_1[i].view(1))
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(y_pred.numpy(), bce_loss_0.numpy(), label='BCE (target=0)')
    plt.plot(y_pred.numpy(), focal_0.numpy(), label='Focal (target=0)')
    plt.xlabel("Prediction")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Comparison of Loss Functions (Target = 0)")
    
    plt.subplot(1, 2, 2)
    plt.plot(y_pred.numpy(), bce_loss_1.numpy(), label='BCE (target=1)')
    plt.plot(y_pred.numpy(), focal_1.numpy(), label='Focal (target=1)')
    plt.xlabel("Prediction")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Comparison of Loss Functions (Target = 1)")
    
    plt.tight_layout()
    plt.show()

# Example usage
# visualize_losses()
```

### 10.5 Transfer Learning

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import time

# Setup device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Load a pretrained model (e.g., ResNet18)
def get_pretrained_model(num_classes=10):
    # Load pretrained ResNet
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the final fully connected layer
    # ResNet18's fc layer has 512 inputs
    model.fc = nn.Linear(512, num_classes)
    
    # Move model to device
    model.to(device)
    
    return model

# Setup data transformations for ImageNet models
def get_transforms():
    # ImageNet normalization values
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    # Training transforms with augmentation
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    # Validation transforms
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    return train_transform, val_transform

# Fine-tuning a pretrained model
def fine_tune_model(model, unfreeze_layers=0):
    """Unfreezes the last 'unfreeze_layers' layers for fine-tuning"""
    
    # Unfreeze the final classifier regardless
    for param in model.fc.parameters():
        param.requires_grad = True
    
    # Unfreeze the specified number of layers from the end
    if unfreeze_layers > 0:
        children = list(model.children())
        for child in children[-unfreeze_layers-1:-1]:  # Exclude final fc layer which is already unfrozen
            for param in child.parameters():
                param.requires_grad = True
    
    return model

# Feature extraction example
def extract_features(model, image_path):
    """Extract features from a specific layer of the model"""
    # Load and preprocess image
    transform = get_transforms()[1]  # Use validation transform
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # Extract features from various layers
    features = {}
    
    def get_activation(name):
        def hook(model, input, output):
            features[name] = output.detach()
        return hook
    
    # Register hooks
    handles = []
    handles.append(model.layer1.register_forward_hook(get_activation('layer1')))
    handles.append(model.layer2.register_forward_hook(get_activation('layer2')))
    handles.append(model.layer3.register_forward_hook(get_activation('layer3')))
    handles.append(model.layer4.register_forward_hook(get_activation('layer4')))
    handles.append(model.avgpool.register_forward_hook(get_activation('avgpool')))
    
    # Forward pass
    with torch.no_grad():
        model(input_tensor)
    
    # Remove hooks
    for handle in handles:
        handle.remove()
    
    return features

# Visualize feature maps
def visualize_feature_maps(features, layer_name='layer1'):
    feature_map = features[layer_name][0]  # First image in batch
    
    # Visualize first 16 feature maps or less
    num_features = min(16, feature_map.size(0))
    
    plt.figure(figsize=(15, 8))
    for i in range(num_features):
        plt.subplot(4, 4, i+1)
        plt.imshow(feature_map[i].cpu().numpy(), cmap='viridis')
        plt.axis('off')
    
    plt.suptitle(f"Feature Maps from {layer_name}")
    plt.tight_layout()
    plt.show()

# Example usage:
# model = get_pretrained_model(num_classes=10)
# model = fine_tune_model(model, unfreeze_layers=2)  # Unfreeze last 2 layers
# 
# # Count trainable parameters
# trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f"Number of trainable parameters: {trainable_params}")
# 
# # Extract features from an image
# features = extract_features(model, 'path/to/image.jpg')
# visualize_feature_maps(features, 'layer1')
```

### 10.6 Model Saving, Loading and Deployment

```python
import torch
import torch.nn as nn
import torch.onnx
import onnx
import time
import io
import numpy as np

# 1. Saving and loading models
class SimpleClassifier(nn.Module):
    def __init__(self, input_size=10, hidden_size=50, output_size=2):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Create model
model = SimpleClassifier()

# Fill with random parameters
dummy_input = torch.randn(1, 10)
dummy_output = model(dummy_input)

# Method 1: Save the entire model
torch.save(model, 'full_model.pth')

# Method 2: Save only the state dict (recommended)
torch.save(model.state_dict(), 'model_state_dict.pth')

# Method 3: Save checkpoint (for resuming training)
checkpoint = {
    'epoch': 10,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': torch.optim.Adam(model.parameters()).state_dict(),
    'loss': 0.1,
    # other items to save
}
torch.save(checkpoint, 'checkpoint.pth')

# Loading the model back
def load_model_examples():
    # Method 1: Load the full model
    loaded_model = torch.load('full_model.pth')
    loaded_model.eval()
    
    # Method 2: Load the state dict into a model instance
    new_model = SimpleClassifier()
    new_model.load_state_dict(torch.load('model_state_dict.pth'))
    new_model.eval()
    
    # Method 3: Load checkpoint to resume training
    checkpoint = torch.load('checkpoint.pth')
    resume_model = SimpleClassifier()
    resume_model.load_state_dict(checkpoint['model_state_dict'])
    optimizer = torch.optim.Adam(resume_model.parameters())
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Resumed from epoch {epoch} with loss {loss}")

# 2. Converting to TorchScript for deployment
def convert_to_torchscript():
    model = SimpleClassifier()
    model.eval()
    
    # Method 1: Tracing (runs a forward pass with example input)
    traced_model = torch.jit.trace(model, dummy_input)
    traced_model.save('traced_model.pt')
    
    # Method 2: Scripting (analyzes the code)
    scripted_model = torch.jit.script(model)
    scripted_model.save('scripted_model.pt')
    
    # Loading TorchScript models
    loaded_traced = torch.jit.load('traced_model.pt')
    output_traced = loaded_traced(dummy_input)
    
    loaded_scripted = torch.jit.load('scripted_model.pt')
    output_scripted = loaded_scripted(dummy_input)
    
    print("Original output:", dummy_output)
    print("Traced model output:", output_traced)
    print("Scripted model output:", output_scripted)

# 3. ONNX export for interoperability
def export_to_onnx():
    model = SimpleClassifier()
    model.eval()
    
    # Export to ONNX
    torch.onnx.export(
        model,               # model to export
        dummy_input,         # example input
        "model.onnx",        # output file
        export_params=True,  # store trained parameters
        opset_version=11,    # ONNX version
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'},
                     'output': {0: 'batch_size'}}
    )
    
    # Verify ONNX model
    onnx_model = onnx.load("model.onnx")
    onnx.checker.check_model(onnx_model)
    print("ONNX model verified!")

# 4. Model optimization and benchmarking
def benchmark_model(model, input_tensor, num_runs=100):
    # Warm-up
    for _ in range(10):
        _ = model(input_tensor)
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_runs):
        _ = model(input_tensor)
    
    end_time = time.time()
    avg_time = (end_time - start_time) / num_runs
    
    return avg_time

# Example usage:
# load_model_examples()
# convert_to_torchscript()
# export_to_onnx()
# 
# # Benchmark original vs TorchScript models
# model = SimpleClassifier().eval()
# traced_model = torch.jit.load('traced_model.pt')
# 
# model_time = benchmark_model(model, dummy_input)
# traced_time = benchmark_model(traced_model, dummy_input)
# 
# print(f"Original model: {model_time*1000:.3f} ms per inference")
# print(f"TorchScript model: {traced_time*1000:.3f} ms per inference")
# print(f"Speedup: {model_time/traced_time:.2f}x")
```

### 10.7 Hooks and Model Introspection

```python
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

# Create a CNN model for demonstration
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.fc = nn.Linear(32 * 7 * 7, 10)
    
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(-1, 32 * 7 * 7)
        x = self.fc(x)
        return x

# 1. Forward hooks to monitor activations
def register_activation_hooks(model):
    activations = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook
    
    # Register hooks for each layer
    hooks = []
    for name, module in model.named_modules():
        if not list(module.children()): # Only leaf modules
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    return activations, hooks

# 2. Backward hooks to monitor gradients
def register_gradient_hooks(model):
    gradients = {}
    
    def hook_fn(name):
        def hook(module, grad_input, grad_output):
            gradients[name] = grad_output[0].detach()
        return hook
    
    # Register hooks for each layer
    hooks = []
    for name, module in model.named_modules():
        if not list(module.children()): # Only leaf modules
            hooks.append(module.register_backward_hook(hook_fn(name)))
    
    return gradients, hooks

# 3. Weight visualization
def visualize_filters(model, layer_name='conv1', figsize=(12, 8)):
    # Get the selected layer
    for name, module in model.named_modules():
        if name == layer_name:
            # Get weights
            weights = module.weight.data.clone()
            
            # Normalize weights for better visualization
            if len(weights.shape) == 4:  # Conv layer
                # Shape: [out_channels, in_channels, kernel_h, kernel_w]
                n_filters = weights.shape[0]
                n_channels = weights.shape[1]
                
                # Create plot grid
                n_cols = min(8, n_filters)
                n_rows = int(np.ceil(n_filters / n_cols))
                
                plt.figure(figsize=figsize)
                for i in range(n_filters):
                    ax = plt.subplot(n_rows, n_cols, i + 1)
                    
                    # Show filters individually for single channel or average across channels
                    if n_channels == 1:
                        filter_img = weights[i, 0].cpu().numpy()
                    else:
                        filter_img = weights[i].mean(dim=0).cpu().numpy()
                    
                    plt.imshow(filter_img, cmap='viridis')
                    ax.set_xticks([])
                    ax.set_yticks([])
                
                plt.suptitle(f'Filters in {layer_name} layer')
                plt.tight_layout()
                plt.show()
                return
    
    print(f"Layer {layer_name} not found or not a convolutional layer")

# 4. Activation visualization
def visualize_activations(activations, layer_name, figsize=(12, 8)):
    if layer_name in activations:
        act = activations[layer_name]
        
        if len(act.shape) == 4:  # Convolutional layer activations
            # Shape: [batch_size, channels, height, width]
            batch_size, n_channels = act.shape[0], act.shape[1]
            
            # Use first image in batch
            act = act[0]
            
            n_cols = min(8, n_channels)
            n_rows = int(np.ceil(n_channels / n_cols))
            
            plt.figure(figsize=figsize)
            for i in range(n_channels):
                if i < n_rows * n_cols:
                    ax = plt.subplot(n_rows, n_cols, i + 1)
                    plt.imshow(act[i].cpu().numpy(), cmap='viridis')
                    ax.set_xticks([])
                    ax.set_yticks([])
            
            plt.suptitle(f'Activations in {layer_name} layer')
            plt.tight_layout()
            plt.show()
        else:
            print(f"Activations for {layer_name} is not a 4D tensor")
    else:
        print(f"No activations found for {layer_name}")

# Example usage
def run_model_inspection_demo():
    # Create model and dummy input
    model = ConvNet()
    dummy_input = torch.randn(1, 1, 28, 28)
    
    # Register hooks
    activations, forward_hooks = register_activation_hooks(model)
    gradients, backward_hooks = register_gradient_hooks(model)
    
    # Forward pass
    output = model(dummy_input)
    
    # Backward pass (need loss and target)
    target = torch.tensor([5])  # Example target
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(output, target)
    loss.backward()
    
    # Visualize filters
    visualize_filters(model, 'conv1')
    
    # Visualize activations
    visualize_activations(activations, 'conv1')
    
    # Clean up hooks
    for hook in forward_hooks + backward_hooks:
        hook.remove()
    
    # Print model architecture summary
    print("\nModel Architecture Summary:")
    for name, module in model.named_modules():
        if not list(module.children()):  # Only leaf modules
            params = sum(p.numel() for p in module.parameters())
            print(f"{name}: {module.__class__.__name__} with {params:,} parameters")

# run_model_inspection_demo()
```

## Conclusion

This tutorial has covered a wide range of PyTorch topics, from the very basics of tensors and autograd to advanced features like custom layers, model deployment, and distributed training. Here's a summary of what we've learned:

1. **PyTorch Basics**: Tensors, operations, and using GPU acceleration
2. **Autograd**: Understanding how automatic differentiation works
3. **Neural Networks**: Building models with nn.Module and nn.Sequential
4. **Training Loop**: Creating effective training pipelines
5. **Classification Tasks**: Binary and multi-class classification problems
6. **Regression**: Predicting continuous values
7. **Optimization Techniques**: Different optimizers, learning rate schedules, and regularization methods
8. **Advanced Features**: Custom datasets, loss functions, transfer learning, and model deployment

### Next Steps

To continue expanding your PyTorch knowledge:

1. Explore more complex architectures (Transformers, GANs, etc.)
2. Dive deeper into specific domains (NLP, Computer Vision, Reinforcement Learning)
3. Experiment with PyTorch Lightning for more structured training loops
4. Learn about quantization and pruning for model optimization
5. Try out distributed training on multiple GPUs or machines
6. Explore PyTorch's C++ frontend for production deployment

Remember that deep learning is as much an art as it is a science. Experiment with different models and hyperparameters, and don't forget to visualize your data and results to gain insights.


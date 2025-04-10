
# PyTorch Linear Regression with Batching Tutorial

This tutorial walks through building and training a simple linear regression model in PyTorch, including support for mini-batch gradient descent.

---

## 1. Setup and Data Generation

We start by importing necessary libraries and generating synthetic data.

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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
```

---

## 2. Creating a Dataset and DataLoader (Mini-Batching)

We wrap the data in a `TensorDataset` and load it using `DataLoader` to enable batching.

```python
dataset = TensorDataset(X_tensor, y_tensor)
batch_size = 16  # Change to 1 for SGD, or 100 for full-batch
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
```

---

## 3. Building the Linear Regression Model

A simple model with a single `nn.Linear` layer.

```python
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)

model = LinearRegressionModel()
```

---

## 4. Defining Loss and Optimizer

Using Mean Squared Error (MSE) loss and SGD optimizer.

```python
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)
```

---

## 5. Training Loop with Mini-Batch Gradient Descent

Training the model using batches from the DataLoader.

```python
num_epochs = 100
losses = []

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for batch_X, batch_y in dataloader:
        y_pred = model(batch_X)
        loss = criterion(y_pred, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / len(dataloader)
    losses.append(avg_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')
```

---

## 6. Visualizing the Results

```python
# Print model parameters
print(f'Weight: {model.linear.weight.item():.4f}, Bias: {model.linear.bias.item():.4f}')

# Plotting
plt.figure(figsize=(10, 6))

# Loss curve
plt.subplot(1, 2, 1)
plt.plot(range(num_epochs), losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')

# Regression result
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

---

## Summary

| Gradient Descent Type | Batch Size | Notes |
|------------------------|------------|-------|
| Stochastic (SGD)       | 1          | High variance updates, fast but noisy |
| Mini-batch             | 16, 32...  | Balance between speed and stability |
| Full-batch             | 100 (all)  | Stable updates, slower |

To experiment, change the `batch_size` value accordingly.

---

Happy experimenting with batching in PyTorch!
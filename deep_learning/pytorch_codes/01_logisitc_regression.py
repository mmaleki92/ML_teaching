import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic binary classification data
X, y = make_classification(n_samples=1000, n_features=2, n_informative=2, 
                           n_redundant=0, n_clusters_per_class=1, random_state=42)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.FloatTensor(y_test).reshape(-1, 1)


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        """
        Logistic Regression model implementation from scratch.
        
        Args:
            input_dim (int): Number of input features
        """
        super(LogisticRegression, self).__init__()
        
        # Weights and bias parameters
        self.weights = nn.Parameter(torch.randn(input_dim, 1) * 0.01)
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        """
        Forward pass of the logistic regression model.
        
        Args:
            x (torch.Tensor): Input features
            
        Returns:
            torch.Tensor: Predicted probabilities
        """
        # Linear transformation: z = xW + b
        z = torch.mm(x, self.weights) + self.bias
        
        # Apply sigmoid activation to get probabilities
        # sigmoid(z) = 1 / (1 + exp(-z))
        probs = 1 / (1 + torch.exp(-z))
        
        return probs
    
    def predict(self, x, threshold=0.5):
        """
        Make binary predictions based on probability threshold.
        
        Args:
            x (torch.Tensor): Input features
            threshold (float): Probability threshold for positive class
            
        Returns:
            torch.Tensor: Binary predictions (0 or 1)
        """
        probs = self.forward(x)
        return (probs >= threshold).float()


# Define binary cross entropy loss function from scratch
def binary_cross_entropy(y_pred, y_true):
    """
    Binary Cross Entropy Loss function implemented from scratch.
    
    BCE = -(y * log(y_pred) + (1 - y) * log(1 - y_pred))
    
    Args:
        y_pred (torch.Tensor): Predicted probabilities
        y_true (torch.Tensor): True binary labels
        
    Returns:
        torch.Tensor: Mean binary cross entropy loss
    """
    # Small epsilon to avoid log(0)
    epsilon = 1e-7
    
    # Clip predictions for numerical stability
    y_pred = torch.clamp(y_pred, epsilon, 1 - epsilon)
    
    # Calculate binary cross entropy loss
    loss = -(y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
    
    # Return mean loss across all samples
    return loss.mean()


# Calculate accuracy
def accuracy(y_pred, y_true, threshold=0.5):
    """
    Calculate classification accuracy.
    
    Args:
        y_pred (torch.Tensor): Predicted probabilities
        y_true (torch.Tensor): True binary labels
        threshold (float): Probability threshold for positive class
        
    Returns:
        float: Accuracy score
    """
    y_pred_class = (y_pred >= threshold).float()
    return (y_pred_class == y_true).float().mean().item() * 100


# Initialize model, optimizer, and hyperparameters
input_dim = X_train.shape[1]
model = LogisticRegression(input_dim)
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Training hyperparameters
num_epochs = 1000
batch_size = 32
num_batches = int(np.ceil(len(X_train) / batch_size))

# Lists to store metrics for plotting
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

# Training loop
for epoch in range(num_epochs):
    # Shuffle indices
    indices = torch.randperm(X_train_tensor.size(0))
    
    epoch_loss = 0.0
    epoch_acc = 0.0
    
    # Mini-batch training
    for i in range(num_batches):
        # Get batch indices
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, X_train_tensor.size(0))
        batch_indices = indices[start_idx:end_idx]
        
        # Get batch data
        X_batch = X_train_tensor[batch_indices]
        y_batch = y_train_tensor[batch_indices]
        
        # Forward pass
        y_pred = model(X_batch)
        
        # Compute loss
        loss = binary_cross_entropy(y_pred, y_batch)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate batch loss and accuracy
        epoch_loss += loss.item() * len(batch_indices)
        epoch_acc += accuracy(y_pred, y_batch) * len(batch_indices)
    
    # Calculate epoch metrics
    epoch_loss /= len(X_train_tensor)
    epoch_acc /= len(X_train_tensor)
    
    # Evaluate on test set
    with torch.no_grad():
        test_preds = model(X_test_tensor)
        test_loss = binary_cross_entropy(test_preds, y_test_tensor).item()
        test_acc = accuracy(test_preds, y_test_tensor)
    
    # Store metrics
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    # Print progress every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}%, "
              f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%")

# Print final model parameters
print("\nFinal Model Parameters:")
print(f"Weights: {model.weights.data.numpy().flatten()}")
print(f"Bias: {model.bias.data.item()}")

# Plot decision boundary
def plot_decision_boundary(model, X, y):
    # Create a mesh grid of points
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Get model predictions for all mesh grid points
    with torch.no_grad():
        Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape).numpy()
    
    # Plot the decision boundary and data points
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')
    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.savefig('decision_boundary.png')
    plt.show()

# Plot learning curves
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.title('Loss vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Binary Cross-Entropy Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Train Accuracy')
plt.plot(test_accuracies, label='Test Accuracy')
plt.title('Accuracy vs. Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()

plt.tight_layout()
plt.savefig('learning_curves.png')
plt.show()

# Plot decision boundary
plot_decision_boundary(model, X_test, y_test)

# Make predictions on new data
def predict_new_data(model, features):
    # Standardize features using the same scaler
    features_scaled = scaler.transform(features)
    features_tensor = torch.FloatTensor(features_scaled)
    
    # Get predictions
    with torch.no_grad():
        probs = model(features_tensor)
        preds = (probs >= 0.5).float()
    
    return preds.numpy(), probs.numpy()

# Example of using the model for new data
new_data = np.array([[1.2, 0.5], [-0.8, -1.2], [2.0, 0.3]])
predictions, probabilities = predict_new_data(model, new_data)

print("\nPredictions for new data:")
for i in range(len(new_data)):
    print(f"Sample {i+1}: Features={new_data[i]}, "
          f"Prediction={predictions[i][0]:.0f}, "
          f"Probability={probabilities[i][0]:.4f}")
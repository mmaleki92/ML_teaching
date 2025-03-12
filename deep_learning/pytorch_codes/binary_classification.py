import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Generate synthetic binary classification data
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=42, n_clusters_per_class=1)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train.reshape(-1, 1))
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test.reshape(-1, 1))

# Define the binary classification model
class BinaryClassifier(nn.Module):
    def __init__(self, input_dim):
        super(BinaryClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 16)
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
input_dim = X_train.shape[1]
model = BinaryClassifier(input_dim)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
batch_size = 32
n_batches = len(X_train) // batch_size
train_losses = []

for epoch in range(num_epochs):
    # Shuffle the data
    indices = torch.randperm(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    epoch_loss = 0
    for i in range(n_batches):
        # Get mini-batch
        start_idx = i * batch_size
        end_idx = start_idx + batch_size
        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]
        
        # Forward pass
        y_pred = model(X_batch)
        
        # Compute loss
        loss = criterion(y_pred, y_batch)
        epoch_loss += loss.item()
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    train_losses.append(epoch_loss / n_batches)
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/n_batches:.4f}')

# Evaluate the model
with torch.no_grad():
    y_pred_prob = model(X_test)
    y_pred = (y_pred_prob >= 0.5).float()
    test_accuracy = (y_pred == y_test).float().mean()
    print(f'Test Accuracy: {test_accuracy.item():.4f}')

# Convert tensors to numpy for sklearn metrics
y_test_np = y_test.numpy().flatten()
y_pred_np = y_pred.numpy().flatten()

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test_np, y_pred_np))

# Plot decision boundary
plt.figure(figsize=(12, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('BCE Loss')

# Plot decision boundary
plt.subplot(1, 2, 2)
# Create a mesh grid on which we will run our model
x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Get predictions for all points in the mesh
Z = model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()]))
Z = (Z >= 0.5).float().detach().numpy()

# Reshape the predictions to the shape of the mesh
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.3)

# Plot the test data
test_indices_0 = y_test_np == 0
test_indices_1 = y_test_np == 1
plt.scatter(X_test[test_indices_0, 0], X_test[test_indices_0, 1], c='blue', label='Class 0')
plt.scatter(X_test[test_indices_1, 0], X_test[test_indices_1, 1], c='red', label='Class 1')
plt.title('Binary Classification Decision Boundary')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()

plt.tight_layout()
plt.savefig('binary_classification_results.png')
plt.show()
from sklearn.datasets import fetch_openml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist.data, mnist.target
X, y = mnist.data, mnist.target
mnist_df = pd.DataFrame(X)
mnist_df['label'] = y

# Save to CSV
mnist_df.to_csv('mnist_dataset.csv', index=False)
# Function to perform PCA
def pca(X, num_components):
    # Standardize the data
    X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

    # Compute the covariance matrix
    covariance_matrix = np.cov(X_standardized, rowvar=False)

    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sort the eigenvectors by decreasing eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, idx]

    # Select the top 'num_components' eigenvectors
    principal_components = eigenvectors[:, :num_components]

    # Project the data onto principal components
    X_pca = np.dot(X_standardized, principal_components)

    return X_pca


# Applying PCA to the MNIST dataset
num_components = 2  # Number of components to keep
mnist_pca = pca(X, num_components)

print(mnist_pca.shape)  # The transformed dataset shape
# Convert labels to integers for color mapping
y_int = y.astype(int)

# Plotting
plt.figure(figsize=(12, 8))
scatter = plt.scatter(mnist_pca[:, 0], mnist_pca[:, 1], c=y_int, cmap='tab10', alpha=0.6)
plt.title('MNIST Data: First Two Principal Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(scatter, label='Digit Label')
plt.show()

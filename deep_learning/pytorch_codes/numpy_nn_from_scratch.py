import numpy as np
import matplotlib.pyplot as plt

class NeuralNetwork:
    """
    A multi-layer neural network implementation for regression problems.
    
    This class implements a fully connected neural network with customizable
    architecture, using only NumPy. It includes forward and backward propagation
    algorithms and trains using Stochastic Gradient Descent (SGD).
    """
    
    def __init__(self, layer_sizes, learning_rate=0.01, epochs=1000, batch_size=32, seed=42):
        """
        Initialize the neural network with specified architecture.
        
        Parameters:
        -----------
        layer_sizes : list
            List of integers specifying the number of neurons in each layer.
            First element is the input size, last element is the output size (typically 1 for regression).
            Elements in between represent the hidden layers.
            Example: [2, 4, 3, 1] creates a network with 2 inputs, 2 hidden layers (4 and 3 neurons), and 1 output.
        learning_rate : float
            Step size for weight updates during gradient descent.
        epochs : int
            Number of complete passes through the training dataset.
        batch_size : int
            Number of samples in each mini-batch for SGD.
        seed : int
            Random seed for reproducibility.
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_layers = len(layer_sizes)
        self.loss_history = []
        np.random.seed(seed)
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # He initialization for weights - helps prevent vanishing/exploding gradients
        for i in range(1, self.n_layers):
            # Scale weights by sqrt(2/n) for ReLU activation (He initialization)
            scale = np.sqrt(2 / layer_sizes[i-1])
            self.weights.append(np.random.randn(layer_sizes[i-1], layer_sizes[i]) * scale)
            self.biases.append(np.zeros((1, layer_sizes[i])))
    
    def relu(self, x):
        """
        ReLU activation function: f(x) = max(0, x)
        Used for hidden layers to introduce non-linearity.
        """
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """
        Derivative of ReLU function:
        f'(x) = 1 if x > 0
        f'(x) = 0 if x <= 0
        """
        return np.where(x > 0, 1.0, 0.0)
    
    def forward(self, X):
        """
        Forward propagation through the network.
        
        Parameters:
        -----------
        X : numpy array
            Input features, shape (batch_size, input_features).
            
        Returns:
        --------
        list of [pre_activations, activations] for each layer
        The final element contains the network's output.
        """
        # Store pre-activations (z) and activations (a) for backpropagation
        cache = []
        
        # Input layer activation is just the input itself
        a = X
        
        # Loop through all layers except the output
        for i in range(self.n_layers - 2):
            # Compute pre-activation: z = a * W + b
            z = np.dot(a, self.weights[i]) + self.biases[i]
            # Apply ReLU activation for hidden layers
            a = self.relu(z)
            # Store for backpropagation
            cache.append((z, a))
        
        # Output layer (no activation for regression - just linear output)
        z = np.dot(a, self.weights[-1]) + self.biases[-1]
        output = z  # Linear activation for regression
        cache.append((z, output))
        
        return cache
    
    def compute_loss(self, y_pred, y_true):
        """
        Compute Mean Squared Error (MSE) loss for regression.
        
        Parameters:
        -----------
        y_pred : numpy array
            Predicted values, shape (batch_size, 1).
        y_true : numpy array
            True target values, shape (batch_size, 1).
            
        Returns:
        --------
        float: MSE loss value
        """
        return np.mean(np.square(y_pred - y_true))
    
    def backward(self, X, y, cache):
        """
        Backward propagation to compute gradients.
        
        Parameters:
        -----------
        X : numpy array
            Input features, shape (batch_size, input_features).
        y : numpy array
            True target values, shape (batch_size, 1).
        cache : list
            Output from forward propagation containing pre-activations and activations.
            
        Returns:
        --------
        gradients of weights and biases
        """
        batch_size = X.shape[0]
        _, activations = zip(*cache)
        
        # Initialize lists to store gradients
        dw = [None] * len(self.weights)
        db = [None] * len(self.biases)
        
        # Get predictions (output of last layer)
        y_pred = activations[-1]
        
        # Output layer error (derivative of MSE with respect to output)
        # For MSE, this is 2(y_pred - y_true)/batch_size simplified to save computation
        error = (y_pred - y) / batch_size
        
        # Backpropagate the error through the network
        for i in reversed(range(len(self.weights))):
            # For output layer (linear activation)
            if i == len(self.weights) - 1:
                # Gradient of weights: dL/dw = a^T * error
                dw[i] = np.dot(activations[i-1].T, error)
                # Gradient of biases: dL/db = sum of errors
                db[i] = np.sum(error, axis=0, keepdims=True)
                # Propagate error to previous layer
                error = np.dot(error, self.weights[i].T)
            else:
                # For hidden layers with ReLU activation
                z, _ = cache[i]
                # Apply derivative of ReLU to error
                error = error * self.relu_derivative(z)
                
                # Input to first hidden layer is X
                prev_activation = X if i == 0 else activations[i-1]
                
                # Compute gradients
                dw[i] = np.dot(prev_activation.T, error)
                db[i] = np.sum(error, axis=0, keepdims=True)
                
                # Propagate error to previous layer (if not at input layer)
                if i > 0:
                    error = np.dot(error, self.weights[i].T)
        
        return dw, db
    
    def update_parameters(self, dw, db):
        """
        Update weights and biases using gradient descent.
        
        Parameters:
        -----------
        dw : list
            Gradients of weights.
        db : list
            Gradients of biases.
        """
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * dw[i]
            self.biases[i] -= self.learning_rate * db[i]
    
    def train(self, X, y, X_val=None, y_val=None):
        """
        Train the neural network using mini-batch SGD.
        
        Parameters:
        -----------
        X : numpy array
            Training features, shape (n_samples, input_features).
        y : numpy array
            Training targets, shape (n_samples, 1).
        X_val : numpy array, optional
            Validation features for monitoring performance.
        y_val : numpy array, optional
            Validation targets for monitoring performance.
            
        Returns:
        --------
        Loss history during training
        """
        n_samples = X.shape[0]
        n_batches = int(np.ceil(n_samples / self.batch_size))
        
        for epoch in range(self.epochs):
            # Shuffle data for each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            # Mini-batch training
            for batch in range(n_batches):
                start_idx = batch * self.batch_size
                end_idx = min(start_idx + self.batch_size, n_samples)
                
                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]
                
                # Forward pass
                cache = self.forward(X_batch)
                
                # Backward pass
                dw, db = self.backward(X_batch, y_batch, cache)
                
                # Update parameters
                self.update_parameters(dw, db)
            
            # Compute and store loss after each epoch
            _, final_activations = zip(*self.forward(X))
            train_loss = self.compute_loss(final_activations[-1], y)
            
            if X_val is not None and y_val is not None:
                _, val_activations = zip(*self.forward(X_val))
                val_loss = self.compute_loss(val_activations[-1], y_val)
                self.loss_history.append((train_loss, val_loss))
                
                # Print progress every 100 epochs
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}/{self.epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            else:
                self.loss_history.append(train_loss)
                
                # Print progress every 100 epochs
                if epoch % 100 == 0:
                    print(f"Epoch {epoch}/{self.epochs}, Loss: {train_loss:.6f}")
        
        return self.loss_history
    
    def predict(self, X):
        """
        Make predictions with the trained network.
        
        Parameters:
        -----------
        X : numpy array
            Input features, shape (n_samples, input_features).
            
        Returns:
        --------
        numpy array: Predictions, shape (n_samples, 1).
        """
        cache = self.forward(X)
        return cache[-1][1]  # Return the activations of the last layer


# Example usage: Creating a simple regression problem
def generate_regression_data(n_samples=1000, noise=0.3, seed=42):
    """Generate synthetic data for regression."""
    np.random.seed(seed)
    X = np.random.uniform(-3, 3, (n_samples, 1))
    # Non-linear function: y = sin(x) + x^2/3 + noise
    y = np.sin(X) + X**2/3 + np.random.normal(0, noise, (n_samples, 1))
    return X, y

def plot_results(model, X, y):
    """Plot the regression results."""
    plt.figure(figsize=(12, 5))
    
    # Plot training history
    plt.subplot(1, 2, 1)
    if isinstance(model.loss_history[0], tuple):
        train_losses, val_losses = zip(*model.loss_history)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
    else:
        plt.plot(model.loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History')
    plt.legend()
    
    # Plot predictions
    plt.subplot(1, 2, 2)
    # Sort X for smooth plotting
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    
    predictions = model.predict(X_sorted)
    
    plt.scatter(X, y, alpha=0.5, label='True data')
    plt.plot(X_sorted, predictions, 'r-', linewidth=2, label='Predictions')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Neural Network Regression')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def main():
    # Generate data
    X, y = generate_regression_data(n_samples=500)
    
    # Split data into training and validation sets
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create and train the model
    # Architecture: 1 input -> 16 neurons -> 8 neurons -> 1 output
    model = NeuralNetwork(
        layer_sizes=[1, 16, 8, 1],  # [input_size, hidden_layers..., output_size]
        learning_rate=0.01,
        epochs=1000,
        batch_size=32
    )
    
    # Train the model
    model.train(X_train, y_train, X_val, y_val)
    
    # Plot results
    plot_results(model, X, y)
    
    # Example prediction
    x_test = np.array([[2.0]])
    pred = model.predict(x_test)
    print(f"Prediction for x=2.0: {pred[0][0]:.4f}")

if __name__ == "__main__":
    main()
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleRNN(nn.Module):
    def __init__(self, n_x, n_a, n_y):
        """
        Initialize the RNN model.
        Arguments:
        n_x -- size of the input layer
        n_a -- size of the hidden layer
        n_y -- size of the output layer
        """
        super(SimpleRNN, self).__init__()
        self.n_a = n_a
        
        # Weight matrices and biases
        self.Wax = nn.Parameter(torch.randn(n_a, n_x) * 0.01)  # Weight for input
        self.Waa = nn.Parameter(torch.randn(n_a, n_a) * 0.01)  # Weight for hidden state
        self.Wya = nn.Parameter(torch.randn(n_y, n_a) * 0.01)  # Weight for output
        self.ba = nn.Parameter(torch.zeros(n_a, 1))  # Bias for hidden state
        self.by = nn.Parameter(torch.zeros(n_y, 1))  # Bias for output

    def forward(self, x_t, a_prev):
        """
        Perform one step of forward propagation for RNN.
        Arguments:
        x_t -- input at time step t, of shape (n_x, batch_size)
        a_prev -- previous hidden state, of shape (n_a, batch_size)
        
        Returns:
        a_next -- next hidden state, of shape (n_a, batch_size)
        y_pred -- output prediction, of shape (n_y, batch_size)
        """
        # Compute next hidden state
        a_next = torch.tanh(torch.matmul(self.Wax, x_t) + torch.matmul(self.Waa, a_prev) + self.ba)
        
        # Compute output
        y_pred = torch.matmul(self.Wya, a_next) + self.by
        
        return a_next, y_pred

def create_one_hot_encoding(text, vocab):
    """
    Create one-hot encoding for the given text based on the vocabulary.
    Arguments:
    text -- input text
    vocab -- dictionary mapping characters to indices

    Returns:
    encoded_text -- list of one-hot encoded vectors
    """
    one_hot_encoding = torch.eye(len(vocab))
    encoded_text = [one_hot_encoding[vocab[char]] for char in text]
    return torch.stack(encoded_text)

def train_rnn(rnn, data, vocab, epochs=100, lr=0.01):
    """
    Train the RNN on the given data.
    Arguments:
    rnn -- SimpleRNN model
    data -- input text data (as a string)
    vocab -- dictionary mapping characters to indices
    epochs -- number of training epochs
    lr -- learning rate
    """
    # Prepare data
    input_text = data[:-1]  # All except the last character
    target_text = data[1:]  # All except the first character

    # Convert to one-hot encoding
    x_train = create_one_hot_encoding(input_text, vocab).T  # Shape: (n_x, seq_len)
    y_train = [vocab[char] for char in target_text]  # Convert target to indices

    # Initialize optimizer and loss
    optimizer = optim.Adam(rnn.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        # Initialize hidden state
        a_prev = torch.zeros(rnn.n_a, 1)

        loss = 0
        for t in range(x_train.shape[1]):
            # Get input and target at time step t
            x_t = x_train[:, t].unsqueeze(1)  # Shape: (n_x, 1)
            y_t = torch.tensor([y_train[t]])  # Shape: (1)

            # Forward pass
            a_prev, y_pred = rnn(x_t, a_prev)

            # Adjust y_pred shape to [batch_size, num_classes]
            # Fix: properly reshape y_pred for cross entropy loss
            y_pred = y_pred.permute(1, 0)  # Shape: (1, n_y)

            # Compute loss
            loss += criterion(y_pred, y_t)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

if __name__ == "__main__":
    # Define text and vocabulary
    text = "hello world"
    vocab = {char: idx for idx, char in enumerate(set(text))}  # Mapping of characters to indices
    print(vocab)
    # Define dimensions
    n_x = len(vocab)  # Size of input (number of unique characters)
    n_a = 10  # Size of hidden state
    n_y = len(vocab)  # Size of output (same as input size for character prediction)

    # Create the RNN
    rnn = SimpleRNN(n_x, n_a, n_y)

    # Train the RNN on the text
    train_rnn(rnn, text, vocab)
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleGRU(nn.Module):
    def __init__(self, n_x, n_h, n_y):
        """
        Initialize the GRU model.
        Arguments:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer
        """
        super(SimpleGRU, self).__init__()
        self.n_h = n_h
        
        # Update gate parameters
        self.Wz = nn.Parameter(torch.randn(n_h, n_x + n_h) * 0.01)
        self.bz = nn.Parameter(torch.zeros(n_h, 1))
        
        # Reset gate parameters
        self.Wr = nn.Parameter(torch.randn(n_h, n_x + n_h) * 0.01)
        self.br = nn.Parameter(torch.zeros(n_h, 1))
        
        # Candidate hidden state parameters
        self.Wh = nn.Parameter(torch.randn(n_h, n_x + n_h) * 0.01)
        self.bh = nn.Parameter(torch.zeros(n_h, 1))
        
        # Output parameters
        self.Wy = nn.Parameter(torch.randn(n_y, n_h) * 0.01)
        self.by = nn.Parameter(torch.zeros(n_y, 1))

    def forward(self, x_t, h_prev):
        """
        Perform one step of forward propagation for GRU.
        Arguments:
        x_t -- input at time step t, of shape (n_x, batch_size)
        h_prev -- previous hidden state, of shape (n_h, batch_size)
        
        Returns:
        h_next -- next hidden state, of shape (n_h, batch_size)
        y_pred -- output prediction, of shape (n_y, batch_size)
        """
        # Concatenate x_t and h_prev
        concat = torch.cat((h_prev, x_t), 0)
        
        # Update gate
        z_t = torch.sigmoid(torch.matmul(self.Wz, concat) + self.bz)
        
        # Reset gate
        r_t = torch.sigmoid(torch.matmul(self.Wr, concat) + self.br)
        
        # Candidate hidden state
        # Concat of (r_t * h_prev) and x_t
        concat_r = torch.cat((r_t * h_prev, x_t), 0)
        h_candidate = torch.tanh(torch.matmul(self.Wh, concat_r) + self.bh)
        
        # New hidden state
        h_next = z_t * h_prev + (1 - z_t) * h_candidate
        
        # Output prediction
        y_pred = torch.matmul(self.Wy, h_next) + self.by
        
        return h_next, y_pred

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

def train_gru(gru, data, vocab, epochs=100, lr=0.01):
    """
    Train the GRU on the given data.
    Arguments:
    gru -- SimpleGRU model
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
    optimizer = optim.Adam(gru.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        # Initialize hidden state
        h_prev = torch.zeros(gru.n_h, 1)

        loss = 0
        for t in range(x_train.shape[1]):
            # Get input and target at time step t
            x_t = x_train[:, t].unsqueeze(1)  # Shape: (n_x, 1)
            y_t = torch.tensor([y_train[t]])  # Shape: (1)

            # Forward pass
            h_prev, y_pred = gru(x_t, h_prev)

            # Adjust y_pred shape to [batch_size, num_classes]
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

def predict_next_char(gru, char, vocab, idx_to_char):
    """
    Predict the next character given an input character
    """
    # Check if character is in vocabulary
    if char not in vocab:
        return "Character not in vocabulary"
    
    # Convert character to one-hot encoding
    x = torch.zeros(len(vocab), 1)
    x[vocab[char]] = 1
    
    # Initialize hidden state
    h = torch.zeros(gru.n_h, 1)
    
    # Forward pass
    _, y_pred = gru(x, h)
    
    # Get the index of the character with highest probability
    predicted_idx = torch.argmax(y_pred).item()
    
    # Convert index to character
    predicted_char = idx_to_char[predicted_idx]
    
    return predicted_char

if __name__ == "__main__":
    # Define text and vocabulary
    text = "hello world"
    chars = sorted(list(set(text)))
    vocab = {char: idx for idx, char in enumerate(chars)}  # Mapping of characters to indices
    idx_to_char = {idx: char for char, idx in vocab.items()}  # Mapping of indices to characters

    # Define dimensions
    n_x = len(vocab)  # Size of input (number of unique characters)
    n_h = 10  # Size of hidden state
    n_y = len(vocab)  # Size of output (same as input size for character prediction)

    # Create the GRU
    gru = SimpleGRU(n_x, n_h, n_y)

    # Train the GRU on the text
    train_gru(gru, text, vocab, epochs=100, lr=0.01)
    
    # Test prediction
    print("\nTesting character prediction:")
    for char in "helo wrd":
        next_char = predict_next_char(gru, char, vocab, idx_to_char)
        print(f"Input: '{char}' → Predicted next: '{next_char}'")
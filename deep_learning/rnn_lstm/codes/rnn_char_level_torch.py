import torch
import torch.nn as nn
import torch.optim as optim

class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initialize the PyTorch RNN model.
        Arguments:
        input_size -- size of the input (vocabulary size)
        hidden_size -- size of the hidden layer
        output_size -- size of the output (vocabulary size)
        """
        super(CharRNN, self).__init__()
        self.hidden_size = hidden_size
        
        # PyTorch's built-in RNN layer
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=False)
        
        # Output layer
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, x, hidden):
        """
        Forward pass through the RNN.
        Arguments:
        x -- input tensor of shape (seq_len, batch_size, input_size)
        hidden -- previous hidden state
        
        Returns:
        output -- output predictions
        hidden -- next hidden state
        """
        # Reshape x if it's just a single time step
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add sequence dimension (1, batch_size, input_size)
            
        # Pass through RNN
        output, hidden = self.rnn(x, hidden)
        
        # Pass through output layer
        output = self.output(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size=1):
        """Initialize hidden state with zeros"""
        return torch.zeros(1, batch_size, self.hidden_size)

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
    rnn -- CharRNN model
    data -- input text data (as a string)
    vocab -- dictionary mapping characters to indices
    epochs -- number of training epochs
    lr -- learning rate
    """
    # Prepare data
    input_text = data[:-1]  # All except the last character
    target_text = data[1:]  # All except the first character

    # Convert to one-hot encoding
    x_train = create_one_hot_encoding(input_text, vocab)  # Shape: (seq_len, n_x)
    x_train = x_train.unsqueeze(1)  # Add batch dimension: (seq_len, batch=1, n_x)
    
    y_train = torch.tensor([vocab[char] for char in target_text])  # Convert target to indices

    # Initialize optimizer and loss
    optimizer = optim.Adam(rnn.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        # Initialize hidden state
        hidden = rnn.init_hidden()

        # Initialize gradients
        optimizer.zero_grad()
        
        # Forward pass through entire sequence
        output, hidden = rnn(x_train, hidden)
        
        # Reshape output for loss calculation
        # output shape: (seq_len, batch=1, n_y) -> (seq_len, n_y)
        output = output.squeeze(1)
        
        # Calculate loss
        loss = criterion(output, y_train)
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        # Print loss
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

def predict_next_char(rnn, char, vocab, idx_to_char):
    """
    Predict the next character given an input character
    """
    # Convert char to one-hot encoding
    if char not in vocab:
        return "Character not in vocabulary"
    
    x = torch.zeros(1, 1, len(vocab))  # (seq_len=1, batch=1, input_size)
    x[0, 0, vocab[char]] = 1
    
    # Initialize hidden state
    hidden = rnn.init_hidden()
    
    # Forward pass
    output, hidden = rnn(x, hidden)
    
    # Get the index of the character with highest probability
    output = output.squeeze()
    _, top_idx = output.topk(1)
    predicted_idx = top_idx[0].item()
    
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
    input_size = len(vocab)  # Size of input (number of unique characters)
    hidden_size = 10  # Size of hidden state
    output_size = len(vocab)  # Size of output (same as input size for character prediction)

    # Create the RNN
    rnn = CharRNN(input_size, hidden_size, output_size)

    # Train the RNN on the text
    train_rnn(rnn, text, vocab, epochs=100, lr=0.01)
    
    # Test prediction
    print("\nTesting character prediction:")
    for char in "helo wrd":
        next_char = predict_next_char(rnn, char, vocab, idx_to_char)
        print(f"Input: '{char}' → Predicted next: '{next_char}'")
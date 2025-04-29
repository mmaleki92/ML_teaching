import torch
import torch.nn as nn
import torch.optim as optim
import re

class SimpleRNN(nn.Module):
    def __init__(self, n_x, n_a, n_y):
        """
        Initialize the RNN model.
        Arguments:
        n_x -- size of the input layer (vocabulary size)
        n_a -- size of the hidden layer
        n_y -- size of the output layer (vocabulary size)
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

def tokenize_paragraph(paragraph):
    """
    Tokenize a paragraph into words.
    Arguments:
    paragraph -- input text paragraph
    
    Returns:
    words -- list of words
    """
    # Clean text and split into words
    paragraph = paragraph.lower()
    words = re.findall(r'\b\w+\b', paragraph)
    return words

def create_word_vocabulary(words):
    """
    Create a vocabulary from words.
    Arguments:
    words -- list of words
    
    Returns:
    vocab -- dictionary mapping words to indices
    idx_to_word -- dictionary mapping indices to words
    """
    # Create vocabulary (unique words)
    unique_words = sorted(list(set(words)))
    vocab = {word: idx for idx, word in enumerate(unique_words)}
    idx_to_word = {idx: word for word, idx in vocab.items()}
    return vocab, idx_to_word

def create_one_hot_encoding(words, vocab):
    """
    Create one-hot encoding for the given words based on the vocabulary.
    Arguments:
    words -- list of words
    vocab -- dictionary mapping words to indices

    Returns:
    encoded_words -- one-hot encoded vectors for words
    """
    one_hot_encoding = torch.eye(len(vocab))
    encoded_words = [one_hot_encoding[vocab[word]] for word in words]
    return torch.stack(encoded_words)

def train_rnn(rnn, paragraph, vocab, idx_to_word, epochs=100, lr=0.01):
    """
    Train the RNN on the given paragraph data.
    Arguments:
    rnn -- SimpleRNN model
    paragraph -- tokenized paragraph (list of words)
    vocab -- dictionary mapping words to indices
    idx_to_word -- dictionary mapping indices to words
    epochs -- number of training epochs
    lr -- learning rate
    """
    # Prepare data
    input_words = paragraph[:-1]  # All words except the last one
    target_words = paragraph[1:]  # All words except the first one

    # Convert to one-hot encoding
    x_train = create_one_hot_encoding(input_words, vocab).T  # Shape: (n_x, seq_len)
    y_train = [vocab[word] for word in target_words]  # Convert target to indices

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

def predict_next_word(rnn, word, vocab, idx_to_word):
    """
    Predict the next word given an input word.
    Arguments:
    rnn -- trained SimpleRNN model
    word -- input word
    vocab -- dictionary mapping words to indices
    idx_to_word -- dictionary mapping indices to words
    
    Returns:
    next_word -- predicted next word
    """
    # Check if word is in vocabulary
    if word not in vocab:
        return "Word not in vocabulary"
    
    # Convert word to one-hot encoding
    x = torch.zeros(len(vocab))
    x[vocab[word]] = 1
    x = x.unsqueeze(1)  # Shape: (n_x, 1)
    
    # Initialize hidden state
    a_prev = torch.zeros(rnn.n_a, 1)
    
    # Forward pass
    _, y_pred = rnn(x, a_prev)
    
    # Get the index of the word with highest probability
    predicted_idx = torch.argmax(y_pred).item()
    
    # Convert index to word
    next_word = idx_to_word[predicted_idx]
    
    return next_word

if __name__ == "__main__":
    # Define a paragraph of text
    paragraph = """
    Natural language processing is a subfield of linguistics, computer science, and artificial intelligence 
    concerned with the interactions between computers and human language, in particular how to program computers 
    to process and analyze large amounts of natural language data. The goal is a computer capable of understanding 
    the contents of documents, including the contextual nuances of the language within them. 
    The technology can then accurately extract information and insights contained in the documents as well as 
    categorize and organize the documents themselves.
    """
    
    # Tokenize paragraph into words
    words = tokenize_paragraph(paragraph)
    
    # Create vocabulary
    vocab, idx_to_word = create_word_vocabulary(words)
    
    # Define dimensions
    n_x = len(vocab)  # Size of input (vocabulary size)
    n_a = 50  # Size of hidden state
    n_y = len(vocab)  # Size of output (same as input size for word prediction)
    
    # Create the RNN
    rnn = SimpleRNN(n_x, n_a, n_y)
    
    # Train the RNN on the paragraph
    print("Training the RNN...")
    train_rnn(rnn, words, vocab, idx_to_word, epochs=100, lr=0.01)
    
    # Example of predicting the next word
    test_words = ["language", "computer", "the", "natural", "process"]
    print("\nPredicting next words:")
    for word in test_words:
        next_word = predict_next_word(rnn, word, vocab, idx_to_word)
        print(f"Input word: '{word}' → Predicted next word: '{next_word}'")
    
    # Interactive mode
    print("\nInteractive mode (type 'exit' to quit):")
    while True:
        word = input("Enter a word: ")
        if word.lower() == 'exit':
            break
        next_word = predict_next_word(rnn, word.lower(), vocab, idx_to_word)
        print(f"Predicted next word: '{next_word}'")
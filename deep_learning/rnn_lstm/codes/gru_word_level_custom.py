import torch
import torch.nn as nn
import torch.optim as optim
import re

class SimpleGRU(nn.Module):
    def __init__(self, n_x, n_h, n_y):
        """
        Initialize the GRU model.
        Arguments:
        n_x -- size of the input layer (vocabulary size)
        n_h -- size of the hidden layer
        n_y -- size of the output layer (vocabulary size)
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

def train_gru(gru, paragraph, vocab, idx_to_word, epochs=100, lr=0.01):
    """
    Train the GRU on the given paragraph data.
    Arguments:
    gru -- SimpleGRU model
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

def predict_next_word(gru, word, vocab, idx_to_word):
    """
    Predict the next word given an input word.
    Arguments:
    gru -- trained SimpleGRU model
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
    x = torch.zeros(len(vocab), 1)
    x[vocab[word]] = 1
    
    # Initialize hidden state
    h = torch.zeros(gru.n_h, 1)
    
    # Forward pass
    _, y_pred = gru(x, h)
    
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
    n_h = 50  # Size of hidden state
    n_y = len(vocab)  # Size of output (same as input size for word prediction)
    
    # Create the GRU
    gru = SimpleGRU(n_x, n_h, n_y)
    
    # Train the GRU on the paragraph
    print("Training the GRU...")
    train_gru(gru, words, vocab, idx_to_word, epochs=100, lr=0.01)
    
    # Example of predicting the next word
    test_words = ["language", "computer", "the", "natural", "process"]
    print("\nPredicting next words:")
    for word in test_words:
        next_word = predict_next_word(gru, word, vocab, idx_to_word)
        print(f"Input word: '{word}' → Predicted next word: '{next_word}'")
    
    # Interactive mode
    print("\nInteractive mode (type 'exit' to quit):")
    while True:
        word = input("Enter a word: ")
        if word.lower() == 'exit':
            break
        next_word = predict_next_word(gru, word.lower(), vocab, idx_to_word)
        print(f"Predicted next word: '{next_word}'")
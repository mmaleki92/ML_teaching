import torch
import torch.nn as nn
import torch.optim as optim
import re
# Add at the beginning of your file
import sys
sys.stdin.reconfigure(encoding='utf-8')
sys.stdout.reconfigure(encoding='utf-8')
class WordGRU(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        """
        Initialize the PyTorch GRU model for word-level prediction.
        Arguments:
        vocab_size -- size of the vocabulary
        hidden_size -- size of the hidden layer
        """
        super(WordGRU, self).__init__()
        self.hidden_size = hidden_size
        
        # Define the GRU layer
        self.gru = nn.GRU(vocab_size, hidden_size, batch_first=False)
        
        # Output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, x, hidden):
        """
        Forward pass through the GRU.
        Arguments:
        x -- input tensor of shape (seq_len, batch_size, vocab_size)
        hidden -- previous hidden state
        
        Returns:
        output -- output predictions
        hidden -- next hidden state
        """
        # Make sure input has 3 dimensions (seq_len, batch_size, vocab_size)
        if x.dim() == 2:
            x = x.unsqueeze(0)  # Add sequence dimension
        
        # Pass through GRU
        output, hidden = self.gru(x, hidden)
        
        # Pass through output layer
        output = self.fc(output)
        
        return output, hidden
    
    def init_hidden(self, batch_size=1):
        """Initialize hidden state with zeros"""
        return torch.zeros(1, batch_size, self.hidden_size)

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
    gru -- WordGRU model
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
    x_train = create_one_hot_encoding(input_words, vocab)  # Shape: (seq_len, vocab_size)
    x_train = x_train.unsqueeze(1)  # Add batch dimension: (seq_len, batch=1, vocab_size)
    
    y_train = torch.tensor([vocab[word] for word in target_words])  # Convert target to indices

    # Initialize optimizer and loss
    optimizer = optim.Adam(gru.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        # Initialize hidden state
        hidden = gru.init_hidden()

        # Initialize gradients
        optimizer.zero_grad()
        
        # Forward pass through entire sequence
        output, _ = gru(x_train, hidden)
        
        # Reshape output for loss calculation
        # output shape: (seq_len, batch=1, vocab_size) -> (seq_len, vocab_size)
        output = output.squeeze(1)
        
        # Calculate loss
        loss = criterion(output, y_train)
        
        # Backpropagation
        loss.backward()
        optimizer.step()

        # Print loss
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

def predict_next_word(gru, word, vocab, idx_to_word):
    """
    Predict the next word given an input word.
    Arguments:
    gru -- trained WordGRU model
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
    x = torch.zeros(1, 1, len(vocab))  # (seq_len=1, batch=1, vocab_size)
    x[0, 0, vocab[word]] = 1
    
    # Initialize hidden state
    hidden = gru.init_hidden()
    
    # Forward pass
    output, _ = gru(x, hidden)
    
    # Get the index of the word with highest probability
    output = output.squeeze()  # Now shape is [vocab_size]
    
    # For a 1D tensor, use max() without dimension
    _, predicted_idx = output.max(0)
    predicted_idx = predicted_idx.item()
    
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
    # Define a paragraph of Persian text
    # paragraph = """
    # پردازش زبان طبیعی یک زیرشاخه از زبان‌شناسی، علوم کامپیوتر و هوش مصنوعی است
    # که با تعاملات بین کامپیوترها و زبان انسانی سروکار دارد، به ویژه اینکه چگونه
    # کامپیوترها را برنامه‌ریزی کنیم تا مقادیر زیادی از داده‌های زبان طبیعی را
    # پردازش و تحلیل کنند. هدف، یک کامپیوتر است که قادر به درک محتوای اسناد،
    # از جمله ظرافت‌های زمینه‌ای زبان موجود در آنها باشد.
    # """
    # Tokenize paragraph into words
    words = tokenize_paragraph(paragraph)
    
    # Create vocabulary
    vocab, idx_to_word = create_word_vocabulary(words)
    
    # Define dimensions
    vocab_size = len(vocab)  # Size of vocabulary
    hidden_size = 50  # Size of hidden state
    
    # Create the GRU
    gru = WordGRU(vocab_size, hidden_size)
    
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
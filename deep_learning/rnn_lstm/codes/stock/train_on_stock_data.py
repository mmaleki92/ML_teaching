import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import os

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return self.X.shape[0]
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class StockGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(StockGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer with dropout
        self.gru = nn.GRU(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward pass through GRU
        out, _ = self.gru(x, h0)
        
        # Take only the output from the last time step
        out = self.fc(out[:, -1, :])
        return out

def add_technical_indicators(data):
    """
    Add technical indicators to the DataFrame
    """
    # Calculate moving averages
    data['MA5'] = data['Close'].rolling(window=5).mean()
    data['MA20'] = data['Close'].rolling(window=20).mean()
    
    # Calculate MACD
    data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
    data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA12'] - data['EMA26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Calculate RSI
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    
    # Avoid division by zero
    loss = loss.replace(0, 1e-10)
    rs = gain / loss
    
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Bollinger Bands
    data['BB_Middle'] = data['Close'].rolling(window=20).mean()
    data['BB_Upper'] = data['BB_Middle'] + 2 * data['Close'].rolling(window=20).std()
    data['BB_Lower'] = data['BB_Middle'] - 2 * data['Close'].rolling(window=20).std()
    
    return data

def load_and_prepare_data(file_path, features=None):
    """
    Load and prepare stock data from CSV file
    """
    # Load data
    print(f"Loading data from {file_path}...")
    data = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print(f"Loaded data with shape: {data.shape}")
    
    # Add technical indicators
    print("Adding technical indicators...")
    data = add_technical_indicators(data)
    
    # Default features if none specified
    if features is None:
        features = ['Open', 'High', 'Low', 'Close', 'Volume', 
                   'MA5', 'MA20', 'MACD', 'RSI', 'BB_Upper', 'BB_Lower']
    
    # Select available features
    available_features = [f for f in features if f in data.columns]
    print(f"Using features: {available_features}")
    
    # Check if we have the Close column which is required for prediction
    if 'Close' not in available_features:
        raise ValueError("'Close' column not found in data but is required for prediction")
    
    # Handle missing data
    data = data[available_features].fillna(method='ffill').fillna(method='bfill')
    
    # Remove any remaining NaN values
    if data.isna().any().any():
        print("Removing rows with NaN values...")
        data = data.dropna()
    
    print(f"Final data shape after preparation: {data.shape}")
    
    # Scale data
    scalers = {}
    scaled_data = pd.DataFrame(index=data.index)
    
    for column in data.columns:
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data[column] = scaler.fit_transform(data[[column]]).flatten()
        scalers[column] = scaler
    
    return scaled_data, scalers, data

def create_sequences(data, seq_length, target_column='Close'):
    """
    Create sequences for training with multiple features
    """
    X, y = [], []
    
    for i in range(len(data) - seq_length):
        X.append(data.iloc[i:i+seq_length].values)
        y.append(data[target_column].iloc[i+seq_length])
    
    return np.array(X), np.array(y).reshape(-1, 1)

def plot_training_history(train_losses, val_losses, ticker):
    """
    Plot training and validation loss history
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'{ticker} - Training History')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{ticker}_training_history.png')
    plt.close()

def plot_predictions(dates, actual, predicted, ticker, title):
    """
    Plot actual vs predicted values
    """
    plt.figure(figsize=(12, 6))
    plt.plot(dates, actual, label='Actual', color='blue')
    plt.plot(dates, predicted, label='Predicted', color='red', alpha=0.7)
    plt.title(f'{ticker} - {title}')
    plt.xlabel('Date')
    plt.ylabel('Price USD')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{ticker}_{title.replace(" ", "_").lower()}.png')
    plt.close()

def train_model(model, train_loader, val_loader, epochs, lr, device, ticker):
    """
    Train the model and return training history
    """
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    early_stop_count = 0
    early_stop_patience = 15
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_count = 0
            # Save best model
            torch.save(model.state_dict(), f'{ticker}_best_model.pth')
        else:
            early_stop_count += 1
            if early_stop_count >= early_stop_patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
                
    return train_losses, val_losses

def predict_future_prices(model, last_sequence, future_days, close_scaler, device):
    """
    Predict future stock prices
    """
    model.eval()
    current_sequence = last_sequence.clone()
    predictions = []
    
    with torch.no_grad():
        for _ in range(future_days):
            # Get prediction for next day
            current_sequence_tensor = current_sequence.unsqueeze(0).to(device)
            prediction = model(current_sequence_tensor).cpu().numpy()[0]
            predictions.append(prediction)
            
            # Update sequence: remove first day and add prediction as last day
            # For the new prediction, we only update the Close price (assuming index 3 is Close)
            # Other features would need to be set differently in real use
            new_last_day = current_sequence[-1].clone()
            new_last_day[3] = torch.tensor(prediction[0])
            
            current_sequence = torch.cat([current_sequence[1:], new_last_day.unsqueeze(0)], dim=0)
            
    # Convert to original scale
    scaled_predictions = np.array(predictions)
    original_predictions = close_scaler.inverse_transform(scaled_predictions)
    
    return original_predictions

def main():
    # Parameters
    ticker = 'AAPL'  # Choose one of the downloaded stocks
    file_path = f"{ticker}_stock_data.csv"
    seq_length = 30
    test_size = 0.15
    val_size = 0.15
    batch_size = 32
    hidden_size = 128
    num_layers = 2
    dropout = 0.2
    epochs = 100
    lr = 0.001
    future_days = 30
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"File {file_path} not found. Please make sure you've downloaded the data.")
        available_files = [f for f in os.listdir() if f.endswith('_stock_data.csv')]
        if available_files:
            print(f"Available stock data files: {available_files}")
            file_path = available_files[0]
            ticker = file_path.split('_')[0]
            print(f"Using {file_path} instead.")
        else:
            print("No stock data files found.")
            return
    
    # Load and prepare data
    scaled_data, scalers, raw_data = load_and_prepare_data(file_path)
    
    # Create sequences
    X, y = create_sequences(scaled_data, seq_length)
    print(f"Created sequences with shapes: X={X.shape}, y={y.shape}")
    
    # Split into train, validation, and test sets
    test_samples = int(len(X) * test_size)
    val_samples = int(len(X) * val_size)
    
    X_test = X[-test_samples:]
    y_test = y[-test_samples:]
    
    X_val = X[-test_samples-val_samples:-test_samples]
    y_val = y[-test_samples-val_samples:-test_samples]
    
    X_train = X[:-test_samples-val_samples]
    y_train = y[:-test_samples-val_samples]
    
    print(f"Data split: Train={X_train.shape}, Val={X_val.shape}, Test={X_test.shape}")
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val)
    
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)
    
    # Create datasets and dataloaders
    train_dataset = StockDataset(X_train_tensor, y_train_tensor)
    val_dataset = StockDataset(X_val_tensor, y_val_tensor)
    test_dataset = StockDataset(X_test_tensor, y_test_tensor)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create and train model
    input_size = X_train.shape[2]  # Number of features
    output_size = 1  # Predicting Close price
    
    model = StockGRU(input_size, hidden_size, num_layers, output_size, dropout)
    model.to(device)
    
    print(f"Training model for {ticker}...")
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, epochs, lr, device, ticker
    )
    
    # Plot training history
    plot_training_history(train_losses, val_losses, ticker)
    
    # Load best model
    model.load_state_dict(torch.load(f'{ticker}_best_model.pth'))
    
    # Evaluate on test data
    model.eval()
    test_predictions = []
    test_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            test_predictions.extend(outputs.cpu().numpy())
            test_targets.extend(y_batch.numpy())
    
    # Scale back to original values
    close_scaler = scalers['Close']
    test_predictions = close_scaler.inverse_transform(np.array(test_predictions))
    test_targets = close_scaler.inverse_transform(np.array(test_targets))
    
    # Plot test predictions
    test_dates = raw_data.index[-len(test_predictions):]
    plot_predictions(test_dates, test_targets, test_predictions, ticker, "Test Predictions")
    
    # Calculate metrics
    mse = ((test_predictions - test_targets) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = np.abs(test_predictions - test_targets).mean()
    
    print(f"\nTest Results for {ticker}:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    
    # Predict future prices
    print(f"\nPredicting future prices for {ticker}...")
    last_sequence = X_test_tensor[-1].to(device)
    future_predictions = predict_future_prices(model, last_sequence, future_days, close_scaler, device)
    
    # Create future dates
    last_date = raw_data.index[-1]
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days)
    
    # Plot future predictions with some historical data
    historical_dates = raw_data.index[-60:]  # Last 60 days of historical data
    historical_prices = raw_data['Close'].values[-60:]
    
    plt.figure(figsize=(14, 7))
    plt.plot(historical_dates, historical_prices, label='Historical Data')
    plt.plot(future_dates, future_predictions, label='Predicted Prices', color='red', linestyle='--')
    plt.title(f'{ticker} - Future Price Predictions')
    plt.xlabel('Date')
    plt.ylabel('Price USD')
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{ticker}_future_predictions.png')
    plt.close()
    
    # Print future predictions
    print(f"\nFuture {ticker} stock prices for the next {future_days} days:")
    for i, (date, price) in enumerate(zip(future_dates, future_predictions)):
        print(f"{date.date()}: ${price[0]:.2f}")

if __name__ == "__main__":
    main()
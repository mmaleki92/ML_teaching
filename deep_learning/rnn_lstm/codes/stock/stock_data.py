import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Install with: pip install yfinance pandas matplotlib

def fetch_stock_data(ticker, period="5y"):
    """
    Fetch historical stock data
    ticker: stock symbol (e.g., 'AAPL', 'MSFT', 'GOOG')
    period: time period to fetch (e.g., '1d', '5d', '1mo', '3mo', '1y', '5y', 'max')
    """
    stock = yf.Ticker(ticker)
    hist = stock.history(period=period)
    return hist

if __name__ == "__main__":
    # Fetch data for a few popular stocks
    tickers = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'TSLA']
    
    # Create empty DataFrame to store all data
    all_data = {}
    
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        data = fetch_stock_data(ticker)
        all_data[ticker] = data['Close']
        
        # Plot the closing prices
        plt.figure(figsize=(12, 6))
        plt.plot(data.index, data['Close'])
        plt.title(f"{ticker} Stock Price")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"{ticker}_stock_price.png")
        
        # Save the data to CSV
        data.to_csv(f"{ticker}_stock_data.csv")
    
    # Combine all stock prices into one DataFrame
    combined_df = pd.DataFrame(all_data)
    combined_df.to_csv("combined_stock_data.csv")
    
    print("Data fetching complete!")
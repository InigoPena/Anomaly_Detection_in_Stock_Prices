import yfinance as yf
import pandas as pd
import numpy as np

msft = yf.Ticker("MSFT")
apple = yf.Ticker("AAPL")
nvidia = yf.Ticker("NVDA")
google = yf.Ticker("GOOGL")
tesla = yf.Ticker("TSLA")
salesforce = yf.Ticker("CRM")

msft_data = msft.history(period="20y")
apple_data = apple.history(period="20y")
nvidia_data = nvidia.history(period="20y")
google_data = google.history(period="20y")
tesla_data = tesla.history(period="20y")
salesforce_data = salesforce.history(period="20y")

tickers = {
    "MSFT": msft_data,
    "AAPL": apple_data,
    "NVDA": nvidia_data,
    "GOOGL": google_data,
    "TSLA": tesla_data,
    "CRM": salesforce_data
}

# Selecting Close, Open, High, Low, Volume
for ticker in tickers:
    tickers[ticker] = tickers[ticker].drop(columns=['Dividends', 'Stock Splits'])

# Normalizing the data
def min_max_scaler(price):
    return (price - price.min()) / (price.max() - price.min())

for ticker in tickers:
    tickers[ticker]['Close_Normalized'] = min_max_scaler(tickers[ticker]['Close'])
    tickers[ticker]['Open_Normalized'] = min_max_scaler(tickers[ticker]['Open'])
    tickers[ticker]['High_Normalized'] = min_max_scaler(tickers[ticker]['High'])
    tickers[ticker]['Low_Normalized'] = min_max_scaler(tickers[ticker]['Low'])
    tickers[ticker]['Volume_Normalized'] = min_max_scaler(tickers[ticker]['Volume'])

for ticker in tickers:
    print(tickers[ticker].head())

# Saving data to CSV
for ticker in tickers:
    tickers[ticker].to_csv(f"data/processed_data/{ticker}_data.csv")
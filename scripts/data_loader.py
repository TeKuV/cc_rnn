import pandas as pd
import yfinance as yf

class DataLoader:
    def __init__(self, ticker="AMD", years=4):
        self.ticker = ticker
        self.years = years
        self.data = None

    def fetch_data(self):
        start_date = pd.Timestamp.today() - pd.DateOffset(years=self.years)
        df = yf.download(self.ticker, start=start_date.strftime("%Y-%m-%d"))
        df = df[["Close"]].dropna().reset_index()
        df.columns = ["Date", "Close"]
        self.data = df
        return df

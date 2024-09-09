import pandas as pd 
import yfinance as yf


if __name__ == '__main__':
    df_spx_cons = pd.read_csv('SPX_constituents.csv')
    symbols_list = df_spx_cons['Symbol'].to_list()
    df = yf.download(symbols_list+['^SPX'], start='2000-01-01', end='2024-08-30')
    df['Adj Close'].to_csv('SPX_components_close_data.csv')
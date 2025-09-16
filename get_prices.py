import time
import os
import pandas as pd
from pathlib import Path
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv

PATH = Path(__file__).parent
INDEX_PATH = PATH/".."/"indexes"

load_dotenv( PATH / "keys.env" )
API_KEY = os.getenv( "API_KEY_ALPHAVANTAGE" )

def get_weekly_data(symbol):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data, meta = ts.get_weekly_adjusted(symbol=symbol)
    print(data)
    df = data[['5. adjusted close']].copy()
    df.columns = ["close"]
    df['symbol'] = symbol
    return df.reset_index()

def get_prices_W( tickers, call_limit ):
    output_name = DATA_PATH/"price_D.csv"

    #carico tickers già scaricati
    old_tickers = []
    if output_name.is_file():
        existing_data = pd.read_csv(output_name)
        old_tickers = existing_data["ticker"].dropna().unique().tolist()
    
    remaining_tickers = [t for t in tickers if t not in old_tickers]    #prendo tutti i ticker dopo quelli già scaricati
    next_tickers = remaining_tickers
    if len(remaining_tickers) > call_limit:
        next_tickers = remaining_tickers[:call_limit]
    data = []
    for i, ticker in enumerate(next_tickers):
        try:
            df = get_weekly_data(ticker)
            print(df)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            df["ticker"] = ticker
            data.append(df[["date", "ticker", "marketCap"]])
            print(f"{ticker} downloaded ({i+1})")
        except Exception as e:
            print(f"Errore su {ticker}: {e}")
        if i < len(next_tickers) - 1:
            time.sleep(12)  # circa 5 richieste/minuto
    
    if data:
        final_df = pd.concat(data)
        if old_tickers:
            final_df.to_csv(output_name, mode='a', header=False)
        else:
            final_df.to_csv(output_name)
        print(f"Dati salvati correttamente")
    else:
        print("Nessun dato da salvare.")

INDEX = "R3000"
YEAR = "24-25"
DATA_PATH = PATH/f"data{INDEX}"/YEAR

if __name__ == "__main__":
    tickers_file_path = INDEX_PATH/INDEX/f"{YEAR}.csv"

    #carico tickers
    tickers = pd.read_csv( tickers_file_path, usecols=["Ticker"] ).iloc[:, 0].dropna().unique().tolist()

    get_prices_W( tickers, 1 )

import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta
import time
from dotenv import load_dotenv
import os

#salvo percorsi utili
PATH = Path(__file__).parent

#carico api key
load_dotenv( PATH / "keys.env" )
API_KEY = os.getenv( "API_KEY_FMP_PREMIUM" )

#scarico market cap giornaliera di tutti i (tickers)
def get_mkt_cap( tickers, pause, output_name="all_mktcap_D.csv" ):
    output_path = DATA_PATH/output_name

    #carico tickers già scaricati
    old_tickers = []
    if output_path.is_file():
        existing_data = pd.read_csv(output_path)
        old_tickers = existing_data["ticker"].dropna().unique().tolist()
    
    remaining_tickers = [t for t in tickers if t not in old_tickers]    #prendo tutti i ticker dopo quelli già scaricati
    
    mkt_cap = []
    for ticker in remaining_tickers:
        api_ticker = ticker.replace(".", "-") #sostituisco eventuali punti
        url = f"{BASE_URL}/historical-market-capitalization/{api_ticker}"
        params = { "apikey": API_KEY }
        try:
            r = requests.get(url, params)
            r.raise_for_status()
            data = r.json()
            if not data:  # se non ci sono dati, skip
                print(f"Nessun dato per {ticker}")
                continue
            df = pd.DataFrame(data)
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            df["ticker"] = ticker
        
            mkt_cap.append(df[["date", "ticker", "marketCap"]])
            print(f"Scaricata mktCap per {ticker}")

        except Exception as e:
            print(f"Errore per {ticker}: {e}")

        time.sleep(pause)
    
    if mkt_cap:
        final_df = pd.concat(mkt_cap, ignore_index=True)
        if old_tickers:
            final_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            final_df.to_csv(output_path, index=False)
        print(f"Dati salvati correttamente")
    else:
        print("Nessun dato da salvare.")

#scarico prezzi giornalieri di tutti i (tickers)
def get_prices( tickers, pause, output_name="all_prices_D.csv" ):
    output_path = DATA_PATH/output_name

    #carico tickers già scaricati
    old_tickers = []
    if output_path.is_file():
        existing_data = pd.read_csv(output_path)
        old_tickers = existing_data["ticker"].dropna().unique().tolist()
    
    remaining_tickers = [t for t in tickers if t not in old_tickers]    #prendo tutti i ticker dopo quelli già scaricati

    prices = []
    for ticker in remaining_tickers:
        api_ticker = ticker.replace(".", "-") #sostituisco eventuali punti
        url = f"{BASE_URL}/historical-price-full/{api_ticker}"
        params = { "apikey": API_KEY }
        try:
            r = requests.get(url, params)
            r.raise_for_status()
            data = r.json()
            if not data:  # se non ci sono dati, skip
                print(f"Nessun dato per {ticker}")
                continue
            df = pd.DataFrame(data["historical"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.sort_values("date")
            df["ticker"] = data["symbol"]
            prices.append(df[["date", "ticker", "close", "volume", "vwap"]])
            print(f"Scaricati prices per {ticker}")

        except Exception as e:
            print(f"Errore per {ticker}: {e}")

        time.sleep(pause)
    
    if prices:
        final_df = pd.concat(prices, ignore_index=True)
        if old_tickers:
            final_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            final_df.to_csv(output_path, index=False)
        print(f"Dati salvati correttamente")
    else:
        print("Nessun dato da salvare.")


def get_news_weekly_counts(tickers, pause, start_date="2020-01-01", output_name="all_news_volume.csv"):
   
    output_path = DATA_PATH / output_name

    #carico tickers già scaricati
    old_tickers = []
    if output_path.is_file():
        existing_data = pd.read_csv(output_path)
        old_tickers = existing_data["ticker"].dropna().unique().tolist()

    remaining_tickers = [t for t in tickers if t not in old_tickers]

    #finestra temporale
    start = pd.to_datetime(start_date).strftime("%Y-%m-%d")
    today = datetime.today().strftime("%Y-%m-%d")

    for ticker in remaining_tickers:
        api_ticker = ticker.replace(".", "-") #sostituisco eventuali punti
        url = f"{BASE_URL}/stock_news"

        #scorro tutte le pagine
        page = 0
        while(True):
            params = {
                "tickers": api_ticker,
                "from": start,
                "to": today,
                "page": page,
                "apikey": API_KEY
            }
            try:
                r = requests.get(url, params)
                r.raise_for_status()
                data = r.json()
                if not data:  # se non ci sono dati, skip
                    print(f"Nessun dato per {ticker}")
                    continue
                df = pd.DataFrame(data)
                df["publishedDate"] = pd.to_datetime(df["publishedDate"])
                weekly = (
                    df.set_index("publishedDate")
                    .resample("W-SUN")
                    .size()
                    .rename("news_count")
                    .reset_index()
                    .rename(columns={"publishedDate": "date"})
                )

                print(df)
                print(weekly)

                time.sleep(1000)

            except Exception as e:
                print(f"Errore per {ticker}: {e}")
        
        time.sleep(pause)


    print(f"Dati news settimanali aggiornati in {output_path}.")
    return remaining_tickers



INDEX = "R3000"
YEAR = "23-24"
DATA_PATH = PATH/"data"/f"data{INDEX}"

BASE_URL = "https://financialmodelingprep.com/api/v3"

if __name__ == "__main__":

    tickers_file_path = DATA_PATH / "all_tickers.csv"
    tickers = pd.read_csv(tickers_file_path)["ticker"].dropna().unique().tolist()
    
    pause = 60 / 300     #max 300 richieste al minuto

    #get_mkt_cap( tickers, pause )

    #get_prices( tickers, pause )

    get_news_weekly_counts( tickers, pause )


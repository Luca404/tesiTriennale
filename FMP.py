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
def get_mkt_cap( tickers, pause, output_name="mktcap_D.csv" ):
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
        print("Dati salvati correttamente")
    else:
        print("Nessun dato da salvare.")

#scarico prezzi giornalieri di tutti i (tickers)
def get_prices( tickers, pause, output_name="prices_D.csv" ):
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
        print("Dati salvati correttamente")
    else:
        print("Nessun dato da salvare.")


def get_news(tickers, pause, start_date="2020-01-01", output_name="news_D.csv"):
   
    output_path = DATA_PATH / output_name

    #carico tickers già scaricati
    old_tickers = []
    if output_path.is_file():
        existing_data = pd.read_csv(output_path)
        old_tickers = existing_data["ticker"].dropna().unique().tolist()

    remaining_tickers = [t for t in tickers if t not in old_tickers]

    #finestra temporale
    start = pd.to_datetime(start_date)
    today = datetime.today().strftime("%Y-%m-%d")

    i = 0
    for ticker in remaining_tickers:
        api_ticker = ticker.replace(".", "-") #sostituisco eventuali punti
        url = f"{BASE_URL}/stock_news"

        data = []
        #scorro tutte le pagine
        page = 0
        print(f"Scaricando dati per {ticker}...")
        while(True):
            params = {
                "tickers": api_ticker,
                "from": start.strftime("%Y-%m-%d"),
                "to": today,
                "page": page,
                "apikey": API_KEY
            }

            r = requests.get(url, params)
            r.raise_for_status()
            page_data = r.json()

            if not page_data or len(page_data) == 0:  #se la pagina è vuota, stop
                break

            for item in page_data:
                date = item.get("publishedDate")
                if not date:
                    continue
                date_time = pd.to_datetime(date, errors="coerce")
                if pd.notna(date_time) and date_time >= start:
                    data.append(item)
            
            page += 1
            time.sleep(pause)

        if not data:
            print(f"Nessun dato per {ticker}")
        else:
            df = pd.DataFrame(data)[["symbol", "publishedDate", "title", "text", "site"]]
            df = df.rename(columns={"symbol": "ticker", "publishedDate": "date"})
            #rimuovo caratteri fastidiosi
            for col in ["text", "title"]:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.replace(r"[\r\n\t]+", " ", regex=True).str.strip()
            print(f"Scaricati dati per {ticker}")
            if old_tickers or i != 0:
                df.to_csv(output_path, mode="a", header=False, index=False )
            else:
                df.to_csv(output_path, index=False )

            i+=1


def get_company_names( tickers, pause, output_name="company_names.csv" ):
    
    output_path = DATA_PATH / output_name

    #carico tickers già scaricati
    old_tickers = []
    if output_path.is_file():
        existing_data = pd.read_csv(output_path)
        old_tickers = existing_data["ticker"].dropna().unique().tolist()

    remaining_tickers = [t for t in tickers if t not in old_tickers]
    
    all_data = []
    for ticker in remaining_tickers:
        api_ticker = ticker.replace(".", "-") #sostituisco eventuali punti
        url = f"{BASE_URL}/profile/{api_ticker}"
        params = { "apikey": API_KEY }
        
        try:
            r = requests.get( url, params=params )
            r.raise_for_status()
            data = r.json()
            if not data:  # se non ci sono dati, skip
                print(f"Nessun dato per {ticker}")
                continue
            all_data.append( {"ticker":ticker, "name":data[0]["companyName"]} )
            print(f"Scaricati dati per {ticker}")
            
        except Exception as e:
            print(f"Errore per {ticker}: {e}")
        
        time.sleep( pause )
        
    if all_data:
        final_df = pd.DataFrame(all_data)
        if old_tickers:
            final_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            final_df.to_csv(output_path, index=False)
        print("Dati salvati correttamente")
    else:
        print("Nessun dato da salvare.")
            
    

INDEX = "MS50"
DATA_PATH = PATH/"data"/INDEX

BASE_URL = "https://financialmodelingprep.com/api/v3"

if __name__ == "__main__":

    tickers_file_path = DATA_PATH / "tickers.csv"
    tickers = pd.read_csv(tickers_file_path)["ticker"].dropna().unique().tolist()

    news_tickers = pd.read_csv( PATH/"data"/"R3000"/"news_volume_D.csv" )["ticker"].dropna().unique().tolist()

    tickers = [t for t in tickers if t not in news_tickers]

    print(tickers)

    pause = 60 / 300     #max 300 richieste al minuto

    #get_mkt_cap( tickers, pause )

    #get_prices( tickers, pause )
    
    get_news( tickers, pause )
    
    #get_company_names( tickers, pause )


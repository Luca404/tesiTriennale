import pandas as pd
import requests
from pathlib import Path
import time
from urllib.parse import quote

PATH = Path(__file__).parent


def get_wiki_views( df, start_date="20200101", output_name="wiki_views_D.csv" ):
    base_url = "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia.org/all-access/all-agents"
    headers = {"User-Agent": "Luca Botta-Thesis/1.0 (contact: luca.botta44@gmail.com)"}
    
    output_path = DATA_PATH / output_name
    #carico tickers già scaricati
    old_tickers = []
    if output_path.is_file():
        existing_data = pd.read_csv(output_path)
        old_tickers = existing_data["ticker"].dropna().unique().tolist()

    remaining_tickers = [t for t in df["ticker"].tolist() if t not in old_tickers]
    
    data = []
    for ticker in remaining_tickers:
        name = df[ df["ticker"] == ticker ]["name"].to_string(index=False)
        url = f"{base_url}/{quote(name, safe='')}/daily/{start_date}/{'20251010'}"
        
        try:
            resp = requests.get(url, headers=headers, timeout=30)

            if resp.status_code == 200:
                items = resp.json().get("items", [])
                for it in items:
                    date = pd.to_datetime(str(it["timestamp"])[:8], format="%Y%m%d", errors="coerce")
                    views = it.get("views", 0)
                    if pd.notna(date):
                        data.append({"date": date, "ticker": ticker, "wiki_views": views})
                print( f"Dati scaricati correttamente per {ticker}" )
            else:
                print(f"errore {resp} per {ticker}")
        
        except Exception as e:
            print(f"Errore per {ticker}: {e}")
        
        time.sleep( 0.5 )
            
    if data:
        out = (pd.DataFrame(data).sort_values(["ticker", "date"]).reset_index(drop=True))
        if old_tickers:
            out.to_csv(output_path, mode='a', header=False, index=False)
        else:
            out.to_csv(output_path, index=False)
        print("Dati salvati correttamente")
        
    else:
        print("Nessun dato da salvare.")

INDEX = "MS50"
DATA_PATH = PATH/"data"/INDEX

if __name__ == "__main__":
    df = pd.read_csv( DATA_PATH / "tickers.csv" )
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    df["name"] = df["name"].astype(str).str.strip()

    get_wiki_views( df )
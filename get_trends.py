#Scarica trends usando DataForSEO API
import os, requests, time, pandas as pd
from datetime import date
from pathlib import Path
from dotenv import load_dotenv
from datetime import date


#salvo percorsi utili
PATH = Path(__file__).parent

#carico api key
load_dotenv( PATH / "keys.env" )
LOGIN = os.getenv("DATAFORSEO_LOGIN")
PASSWORD = os.getenv("DATAFORSEO_PASSWORD")

def get_trends( tickers, output_name="trends_W.csv" ):
    output_path = DATA_PATH / output_name

    #carico tickers già scaricati
    old_tickers = []
    if output_path.is_file():
        existing_data = pd.read_csv(output_path)
        old_tickers = existing_data["ticker"].dropna().unique().tolist()

    remaining_tickers = [t for t in tickers if t not in old_tickers]

    date_from = f"{date.today().year-5}-{date.today().month:02d}-{date.today().day:02d}"
    date_to = date.today().isoformat()
    
    url = "https://api.dataforseo.com/v3/keywords_data/google_trends/explore/live"

    for i in range(0, len(remaining_tickers), 5):
        batch = remaining_tickers[i:i+5]
        payload = [{
            "location_name": "United States",
            "date_from": date_from,
            "date_to": date_to,
            "keywords": batch,
            "type": "web"
        }]
        try:
            r = requests.post(url, auth=(LOGIN, PASSWORD), json=payload, timeout=60)
            r.raise_for_status()
            res = r.json()

            tasks = res.get("tasks") or []
            if not tasks or not tasks[0].get("result"):
                print( res )
                print(f"Nessun risultato per batch {batch}")
                time.sleep(1000)
                continue
            
            result = tasks[0]["result"]
            
            graph = None
            for blk in (result or []):
                if isinstance(blk, dict) and "items" in blk:
                    for it in blk["items"] or []:
                        if it.get("type") == "google_trends_graph" and "data" in it:
                            graph = it["data"]
                            kws = it["keywords"]
                            break
                if graph: break
            
            if not graph:
                continue

            rows = []
            for p in graph:
                # per finestra 5 anni: ogni punto ≈ settimana
                ts = p.get("timestamp")
                dt = pd.to_datetime(ts, unit="s")
                vals = p.get("values") or []
                row = {"date": dt}
                for j, kw in enumerate(kws):
                    if j < len(vals):
                        row[kw] = vals[j]
                rows.append(row)
            
            print(f"{batch} scaricato correttamente")
            
            wide = pd.DataFrame(rows)
            wide["date"] = pd.to_datetime(wide["date"]).dt.tz_localize(None)
            wide = wide.sort_values("date")
            wide = wide.fillna(0)
            long = wide.melt(
                id_vars=["date"],
                var_name="ticker",
                value_name="trend_score"
            ).dropna(subset=["trend_score"])
            
            if old_tickers or i!=0:
                long.to_csv(output_path, mode='a', header=False, index=False)
            else:
                long.to_csv(output_path, index=False)
                
            time.sleep(2.5)

        except Exception as e:
            print(f"Errore con batch {batch}: {e}")
            time.sleep(5)


INDEX = "R3000"
DATA_PATH = PATH/"data"/INDEX

if __name__ == "__main__":
    tickers_file_path = DATA_PATH / "tickers_filtered.csv"
    tickers = pd.read_csv(tickers_file_path)["ticker"].dropna().unique().tolist()
    tickers = tickers[:500]
    get_trends( tickers )
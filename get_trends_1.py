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

def get_trends( tickers, start_date="2020-01-01", output_name="trends_W.csv" ):
    output_path = DATA_PATH / output_name

    #carico tickers già scaricati
    old_tickers = []
    if output_path.is_file():
        existing_data = pd.read_csv(output_path)
        old_tickers = existing_data["ticker"].dropna().unique().tolist()

    remaining_tickers = [t for t in tickers if t not in old_tickers]

    url = "https://api.dataforseo.com/v3/keywords_data/google_trends/explore/live"

    data = []
    for i in range(0, len(remaining_tickers), 5):
        batch = remaining_tickers[i:i+5]
        payload = [{
            "location_name": "United States",
            "date_from": start_date,
            "date_to":date.today().isoformat(),
            "keywords": batch,
            "type": "web"
        }]
        try:
            r = requests.post(url, auth=(LOGIN, PASSWORD), json=payload, timeout=60)
            r.raise_for_status()
            res = r.json()

            tasks = res.get("tasks") or []
            if not tasks or not tasks[0].get("result"):
                print(f"Nessun risultato per batch {batch}")
                continue
            
            result = tasks[0]["result"]
            timeline = None
            for block in result:
                if "interest_over_time" in block:
                    timeline = block["interest_over_time"]["timeline_data"]
                    break
            if not timeline:
                print(f"Nessuna timeline per batch {batch}")
                continue

            rows = []
            for p in timeline:
                t = p.get("time") or p.get("date")
                vals = p.get("values") or []
                if not t:
                    continue
                row = {"date": t}
                for j, kw in enumerate(batch):
                    if j < len(vals):
                        v = vals[j]["value"] if isinstance(vals[j], dict) and "value" in vals[j] else vals[j]
                        row[kw] = v
                rows.append(row)

            df = pd.DataFrame(rows)
            df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            df = df.set_index("date").sort_index()
            data.append(df)

            time.sleep(2.5)

        except Exception as e:
            print(f"Errore con batch {batch}: {e}")
            time.sleep(5)
    
        
    if data:
        final_df = pd.concat(data, axis=1)
        if old_tickers:
            final_df.to_csv(output_path, mode='a', header=False, index=False)
        else:
            final_df.to_csv(output_path, index=False)
        print("Dati salvati correttamente")
    else:
        print("Nessun dato da salvare.")




INDEX = "MS50"
DATA_PATH = PATH/"data"/INDEX

if __name__ == "__main__":
    tickers_file_path = DATA_PATH / "tickers.csv"
    tickers = pd.read_csv(tickers_file_path)["ticker"].dropna().unique().tolist()
    
    get_trends( tickers )
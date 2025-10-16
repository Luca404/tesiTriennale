from pathlib import Path
import pandas as pd
from pytrends.request import TrendReq
import random, time, os
from dotenv import load_dotenv
import requests
import warnings
warnings.simplefilter("ignore", FutureWarning)

user_agents = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
    "Mozilla/5.0 (X11; Linux x86_64)",
]

PATH = Path(__file__).parent
load_dotenv( PATH / "keys.env" )
COOKIES = os.getenv("TRENDS_COOKIES")

def get_google_trends(tickers, tf, output_name="all_trends_W.csv"):
    output_path = DATA_PATH / output_name
    #carico tickers già scaricati
    old_tickers = []
    if output_path.is_file():
        existing_data = pd.read_csv(output_path)
        old_tickers = existing_data["ticker"].dropna().unique().tolist()
        
    remaining_tickers = [t for t in tickers if t not in old_tickers]    #prendo tutti i ticker dopo quelli già scaricati

    ua = random.choice(user_agents)
    pytrends = TrendReq(hl='en-US', tz=360, requests_args={"headers": {"User-Agent":ua, "Cookie": COOKIES}})
    i = 0
    for ticker in remaining_tickers:
        try:
            #richiesta, cat=categoria, geo=stato, gprop=tipo (news, youtube, images, ...)
            pytrends.build_payload([ticker], cat=0, timeframe=tf)
            interest = pytrends.interest_over_time().infer_objects(copy=False)
            
            if not interest.empty:
                print( f"Scaricato {ticker}" )
                df = pd.DataFrame(interest)
                if "isPartial" in df.columns:
                    df = df.drop(columns=["isPartial"])
                
                df_long = df.reset_index().melt(
                    id_vars="date", 
                    var_name="ticker",   
                    value_name="trend"   
                )
                df_long["date"] = pd.to_datetime(df_long["date"])
                
                if old_tickers or i != 0:
                    df_long.to_csv(output_path, mode="a", header=False, index=False )
                else:
                    df_long.to_csv(output_path, index=False )

                i+=1

        except Exception as e:
            if "429" in str(e):
                print(f"Errore 429 per {ticker}. Attendo 10 minuti...")
                time.sleep(600)
                #rigenera la sessione con nuovo UA
                ua = random.choice(user_agents)
                pytrends = TrendReq(hl='en-US', tz=360, requests_args={"headers": {"User-Agent":ua, "Cookie": COOKIES}})
            else:
                print(f"Errore con {ticker}: {e}")
                
        delay = random.uniform(30, 40)
        time.sleep(delay)  #rispetta i limiti di Google

INDEX = "R3000"
DATA_PATH = PATH/"data"/f"data{INDEX}"


if __name__ == "__main__":
    #carico i tickers
    file_name = "all_tickers_filtered.csv"
    tickers = pd.read_csv( DATA_PATH/file_name, usecols=["ticker"] ).iloc[:, 0].dropna().unique().tolist()

    timeframe = "today 5-y"
    
    tickers = tickers[:100]

    get_google_trends(tickers, timeframe)
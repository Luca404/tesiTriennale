from pathlib import Path
import pandas as pd
from pytrends.request import TrendReq
import time
import warnings
warnings.simplefilter("ignore", FutureWarning)


def get_google_trends(tickers, tf, output_name="all_trends_W.csv"):
    output_path = DATA_PATH / output_name
    #carico tickers già scaricati
    old_tickers = []
    if output_path.is_file():
        existing_data = pd.read_csv(output_path)
        old_tickers = existing_data.columns.tolist()
    
    remaining_tickers = [t for t in tickers if t not in old_tickers]    #prendo tutti i ticker dopo quelli già scaricati

    pytrends = TrendReq(hl='en-US', tz=360)
    data = []
    for ticker in remaining_tickers:
        try:
            #richiesta, cat=categoria, geo=stato, gprop=tipo (news, youtube, images, ...)
            pytrends.build_payload([ticker], cat=0, timeframe=tf, geo="US", gprop="")
            interest = pytrends.interest_over_time().infer_objects(copy=False)
            
            if not interest.empty:
                print( f"Scaricato {ticker}" )
                interest = interest[[ticker]]
                data.append(interest)

        except Exception as e:
            print(f"Errore con {ticker}: {e}")
        
        time.sleep(5)  #rispetta i limiti di Google

    #se ci sono dati concateno e salvo
    if data:
        df = pd.concat(data, axis=1)
        if old_tickers:
            df.to_csv(output_path, mode="a", header=False, index=False )
        else:
            df.to_csv(output_path, index=False )

        print("Salvataggio completato")
    else:
        print("Nessun dato disponibile")


PATH = Path(__file__).parent
INDEX = "R3000"
DATA_PATH = PATH/"data"/f"data{INDEX}"


if __name__ == "__main__":
    #carico i tickers
    file_name = "all_tickers.csv"
    tickers = pd.read_csv( DATA_PATH/file_name, usecols=["ticker"] ).iloc[:, 0].dropna().unique().tolist()

    timeframe = "today 5-y"

    tickers = tickers[:200]
    get_google_trends(tickers, timeframe)
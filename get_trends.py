from pathlib import Path
import pandas as pd
from pytrends.request import TrendReq
import time
import warnings
warnings.simplefilter("ignore", FutureWarning)


def get_google_trends(tickers, tf):
    pytrends = TrendReq(hl='en-US', tz=360)
    data = []
    for ticker in tickers:
        try:
            #richiesta, cat=categoria, geo=stato, gprop=tipo (news, youtube, images, ...)
            pytrends.build_payload([ticker], cat=0, timeframe=tf, geo="US", gprop="")
            interest = pytrends.interest_over_time().infer_objects(copy=False)
            
            if not interest.empty:
                print( f"Scaricato {ticker}" )
                interest = interest[[ticker]]
                data.append(interest)
            
            time.sleep(1)  #rispetta i limiti di Google

        except Exception as e:
            print(f"Errore con {ticker}: {e}")

    #se ci sono dati concateno e salvo
    if data:
        df = pd.concat(data, axis=1)
        df.to_csv(DATA_PATH/"all_trends.csv")
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
    get_google_trends(tickers, timeframe)
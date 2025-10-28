# Meme Stock Clustering
Questo progetto nasce come tesi triennale presso l'Università degli Studi di Milano-Bicocca.  
L'obiettivo è analizzare il comportamento delle **meme stocks** all'interno del Russell 3000, applicando tecniche di **clustering** a un insieme di variabili di mercato e di interesse pubblico.

## Obiettivo del lavoro
Il progetto mira a verificare se le azioni note come meme stock presentano caratteristiche comuni che le rendono riconoscibili anche senza etichette predefinite. Come meme stock storiche si considerano le componenti del MS50.
Attraverso l'analisi di più indicatori (close, shorts positions, days-to-cover, Google Trends, news volume, Wikipedia pageviews), il modello cerca di individuare gruppi di titoli con dinamiche simili e di capire se le meme stocks storiche si concentrano in un cluster specifico.

## Metodo
L'analisi si basa sull'algoritmo **TimeSeriesKMeans** con **distanza Dynamic Time Warping (DTW)**, adatto al confronto tra serie temporali di diversa lunghezza o fase.  
Le serie sono standardizzate e organizzate in un dataset multivariato con frequenza bimensile, ancorata alle date FINRA.

**Fasi principali:**
1. Raccolta dati da API (FMP, FINRA, DataForSEO, Wikimedia)
2. Uniformazione delle frequenze e pulizia del dataset
3. Standardizzazione e costruzione delle serie temporali
4. Clustering con DTW per k da 2 a 10
5. Analisi dei risultati tramite Silhouette Score, Davies–Bouldin Index e Adjusted Rand Index

Con **k = 3** il modello individua un cluster che raccoglie circa **l'88% delle meme stocks** considerate, mostrando la capacità del DTW di cogliere pattern comuni.

## Dataset
Il dataset finale è organizzato in formato panel, con osservazioni bimensili per ciascun titolo. Si trova in data/ ed è diviso in due: dataset_meme.csv (contiene le 25 meme stock storiche) e dataset_no-meme.csv (contiene 496 stock generiche)
Le principali variabili incluse sono:

| Colonna | Descrizione |
|----------|-------------|
| close | Prezzo di chiusura a fine finestra |
| d2c | Days to cover (Numero di posizioni corte / Volume medio giornaliero) -> indica quanti giorni ci vorrebbero per chiudere tutte le posizioni corte, se alto indica vulnerabilità ad un potenziale short squeeze |
| shorts | Posizioni corte (Numero totale di posizioni corte aperte) |
| volume | Volume medio giornaliero |
| trend_score | Media bimensile dello score Google Trends |
| news_volume | Numero di articoli nel periodo |
| wiki_views | Numero di visualizzazioni Wikipedia nel periodo |

Campione finale: circa 525 società del Russell 3000, di cui 25 considerate meme stock storiche 
Periodo: 2020–2025


## Struttura della repository
- **FMP.py** – scarica prezzi, capitalizzazioni e news da API FMP
- **extract_news_count.ipynb** - conta le news giornaliere per ogni ticker e salva in news_volume_D.csv
- **get_short_positions.py** – scarica short interest e days-to-cover da FINRA
- **clear_short_data.ipynb** - pulisce short_interest_raw.csv estraendo ticker, d2c, shorts e volume
- **get_trends** - scarica dati di Google Trends con DataForSEO
- **get_wiki_views** - scarica visualizzazioni delle pagine Wikipedia da Wikimedia
- **filter_tickers.ipynb** - filtra le stocks per capitalizzazione, volume e presenza di dati
- **clustering.py** – applica il modello di clustering DTW  
- **merge_data.ipynb** – unisce e pulisce i dataset  
- **result.txt** – contiene i risultati del clustering per diversi k  
- **/data/** – directory con i file CSV e i dataset intermedi
- **Luca_Botta_tesiTriennale.pdf** – testo completo della tesi 

## Sviluppi futuri
Possibili estensioni per lavori successivi:
- Integrare variabili di sentiment (es. numero di post su X, Reddit, ecc.)
- Ridurre l'orizzonte temporale intorno ai periodi in cui si sono verificate le principali ondate di meme stocks (es. 2020–2021), per limitare il rumore
- Aggiungere indicatori fondamentali (P/E, EPS, ecc.)
- Confrontare con altri metodi di clustering (HDBSCAN, K-Shape, GMM)
- Automatizzare la pipeline per aggiornamenti periodici
- Sperimentare un approccio predittivo per stimare la “memeness” di nuovi titoli

## Riferimento
Luca Botta (2025)  
*Tesi triennale: Ranking del Russell 3000 attraverso metodi di clustering per l'individuazione di meme stocks*  
Università degli Studi di Milano-Bicocca  
Relatrice: Prof.ssa Paola Agnese Bongini

from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
import pandas as pd
import os


def read_data(columns, file_name, max_length=None):
    #leggo il file
    file = pd.read_csv(f"data/{file_name}", usecols=["ticker"] + columns)
    #raggruppo per ticker
    groups = file.groupby("ticker", sort=False)
    #calcolo lunghezza massima
    if max_length is None:
        max_length = groups.size().max()
    #definisco array
    X = np.zeros((len(groups), max_length, len(columns)), dtype=np.float32)
    #riempio array
    for i, (ticker, group) in enumerate(groups):
        vals = group[columns].values
        X[i, :len(vals), :] = vals
    #return dataset
    return to_time_series_dataset(X), np.array(list(groups.groups.keys()))

def get_cluster(X, n_clusters, force_create=False):
    filename = f"models/tskm_model__c_{n_clusters}.joblib"
    if not force_create and os.path.exists(filename):
        model = load(filename)
    else:
        print("Nessun modello salvato, clustering...")
        #clustering DTW
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", n_jobs=-1, max_iter=10, random_state=42)
        model.fit_predict(X)
        dump(model, filename)
        print("Clustering fatto")
    return model

def get_distance_matrix(X):
    filename = "data/distance_matrix.joblib"
    if os.path.exists(filename):
        dist_matrix = load(filename)
    else:
        print(f"Distance matrix non trovata, calcolo {X.shape[0] * (X.shape[0] + 1) // 2}...")
        dist_matrix = cdist_dtw(X, n_jobs=-1, verbose=1)
        dump(dist_matrix, filename)
    return dist_matrix

def davies_bouldin_index(X, labels, centroids, max_samples_per_cluster=150, random_state=42):
    rng = np.random.default_rng(random_state)
    K = centroids.shape[0]
    S = np.zeros(K, dtype=np.float64)

    # Coesione intra-cluster S_i
    for i in range(K):
        idx = np.where(labels == i)[0]
        if idx.size == 0:
            S[i] = 0.0
            continue
        # subsampling per velocità
        if idx.size > max_samples_per_cluster:
            idx = rng.choice(idx, size=max_samples_per_cluster, replace=False)
        Xi = X[idx]                                  # (m, T, D)
        Ci = centroids[i][None, ...]                 # (1, T, D)
        # distanze DTW di ciascuna serie del cluster al suo centroide
        dists = cdist_dtw(Xi, Ci).ravel()            # (m,)
        S[i] = np.mean(dists)

    # Separazione tra centroidi M_ij (matrice KxK)
    M = cdist_dtw(centroids, centroids)             # DTW tra centroidi
    # Evita divisioni per zero/diagonale
    np.fill_diagonal(M, np.inf)

    # R_ij = (S_i + S_j) / M_ij
    # Per ogni i, prendi il massimo su j != i
    R = np.zeros(K, dtype=np.float64)
    for i in range(K):
        R[i] = np.max((S[i] + S) / M[i])

    DBI = np.mean(R)
    return DBI

def clustering_stability_ari(X, n_clusters=3, n_bootstrap=5, sample_fraction=0.8, random_state=42):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]
    labels_list = []

    for b in range(n_bootstrap):
        idx = rng.choice(n, size=int(sample_fraction * n), replace=False)
        X_sample = X[idx]

        # Fit solo sul sotto-campione
        model = TimeSeriesKMeans(
            n_clusters=n_clusters,
            metric="dtw",
            n_init=2,        
            max_iter=50,
            random_state=int(rng.integers(0, 1_000_000_000))
        )
        labels_pred = model.fit_predict(X_sample)   #shape = len(idx)

        full_labels = np.full(n, -1)
        full_labels[idx] = labels_pred
        labels_list.append(full_labels)

    ari_scores = []
    for i in range(len(labels_list)):
        for j in range(i + 1, len(labels_list)):
            mask = (labels_list[i] != -1) & (labels_list[j] != -1)
            if mask.sum() > 0:
                ari_scores.append(adjusted_rand_score(labels_list[i][mask], labels_list[j][mask]))

    return float(np.mean(ari_scores)), ari_scores


def main():
    # se true, run the cycle and fit with all cluster in [2,10]
    fit_all = False
    #feature da usare
    features = ['close', 'd2c', 'shorts', 'volume', 'trend_score', 'news_volume', 'wiki_views']
    #numero di cluster da visualizzare
    n_clusters = 3

    #carico i dati
    print("Reading data...")
    X_no_meme, ticker_no_meme = read_data(columns=features, file_name="dataset_no-meme.csv")
    X_meme, ticker_meme = read_data(columns=features, file_name="dataset_meme.csv", max_length=123)
    #unisco i dataset
    X = np.vstack([X_meme, X_no_meme])
    ticker = np.hstack([ticker_meme, ticker_no_meme])
    #Normalizzazione
    X = TimeSeriesScalerMeanVariance().fit_transform(X)

    # Fit models
    if fit_all:
        for c in range(2, 11):
            print(f"Fitting model with n_clusters={c}")
            get_cluster(X, c, force_create=True)


    # Analisi Silhouette score
    print("Silhouette analysis...")
    # calcolo distance matrix
    dist_matrix = get_distance_matrix(X)
    # Calcolo score
    k_values = np.arange(2, 11)
    scores = np.zeros(k_values.shape[0])
    for i, k in enumerate(k_values):
        model = get_cluster(X, k)
        labels = model.labels_
        score = silhouette_score(dist_matrix, labels, metric="precomputed")
        scores[i] = score
        print(f"Silhouette score with k={k}: {score:.3f}")
    # Stampo silhouette score
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, scores, marker='o', linestyle='-', linewidth=2)
    plt.title('Silhouette Score vs Numero di cluster', fontsize=12)
    plt.xlabel('Numero di cluster (k)')
    plt.ylabel('Silhouette score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"img/silhouette_vs_k_{n_clusters}.png", dpi=150)
    plt.show()
    
    
    print("\nCalcolo Davies–Bouldin Index...")
    dbi_values = []
    for k in k_values:
        model_k = get_cluster(X, k)
        labels_k = model_k.labels_
        dbi_k = davies_bouldin_index(
            X=X,
            labels=labels_k,
            centroids=model_k.cluster_centers_,
            max_samples_per_cluster=150, 
            random_state=42
        )
        dbi_values.append(dbi_k)
        print(f"DBI (DTW) con k={k}: {dbi_k:.4f}")

    # Plot DBI vs K
    plt.figure(figsize=(7,4))
    plt.plot(k_values, dbi_values, marker='o', linestyle='-', linewidth=2)
    plt.title("Davies–Bouldin Index vs Numero di cluster", fontsize=12)
    plt.xlabel('Numero di cluster (k)')
    plt.ylabel("DBI")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("img/dbi_vs_k.png", dpi=150)
    plt.show()

    
    '''
    print("\nCalcolo Adjusted Rand Index (ARI) vs numero di cluster...")
    ari_values = []
    for k in k_values:
        mean_ari, ari_scores = clustering_stability_ari(X, n_clusters=k, n_bootstrap=5, sample_fraction=0.8)
        ari_values.append(mean_ari)
        print(f"ARI medio con k={k}: {mean_ari:.3f}  (var={np.std(ari_scores):.3f})")

    # --- Grafico ARI vs numero di cluster ---
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, ari_values, marker='o', linestyle='-', linewidth=2)
    plt.title("Adjusted Rand Index vs numero di cluster", fontsize=12)
    plt.xlabel("Numero di cluster (k)")
    plt.ylabel("ARI medio")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("img/ari_vs_k.png", dpi=150)
    plt.show()
    '''
    

    # Analisi con diversi numeri di cluster
    print(f"Analysis with {n_clusters} clusters...")
    # carico modello
    model = get_cluster(X, n_clusters)
    labels = model.labels_
    
    '''    # Stampo cluster prediction
    for c in range(n_clusters):
        ticker_in_cluster = ticker[labels == c]
        print(f"Dimension of cluster {c+1}: {ticker_in_cluster.shape[0]}, "
              f"meme stock in cluster {c+1}: {sum([t in ticker_meme for t in ticker_in_cluster])}")

    # Visualizzazione
    for c in range(n_clusters):
        plt.figure(figsize=(10, 3))
        plt.suptitle(f"Cluster {c}")
        for j, feature in enumerate(features):
            plt.subplot(1, len(features), j + 1)
            # Cluster series
            for ts in X[labels == c]:
                plt.plot(ts[:, j], 'k-', alpha=0.2)
            # Centroid
            plt.plot(model.cluster_centers_[c][:, j], 'r-', linewidth=2)
            # Title
            plt.title(f"{feature}")
            plt.xlabel("Time")
            plt.tight_layout()
        if c < n_clusters - 1:
            plt.ion()
        else:
            plt.ioff()
        plt.show()
    '''
    

if __name__ == '__main__':
    main()

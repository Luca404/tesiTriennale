from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import silhouette_score
from tslearn.clustering import TimeSeriesKMeans
from tslearn.metrics import cdist_dtw
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
import pandas as pd
import os


def read_data(columns, file_name, max_length=None):
    # Read file
    file = pd.read_csv(f"data/{file_name}", usecols=["ticker"] + columns)
    # Group by ticker
    groups = file.groupby("ticker", sort=False)
    # Compute maximum length
    if max_length is None:
        max_length = groups.size().max()
    # Define array
    X = np.zeros((len(groups), max_length, len(columns)), dtype=np.float32)
    # Fill array
    for i, (ticker, group) in enumerate(groups):
        vals = group[columns].values
        X[i, :len(vals), :] = vals
    # Return dataset
    return to_time_series_dataset(X), np.array(list(groups.groups.keys()))

def get_cluster(X, n_clusters, force_create=False):
    filename = f"models/tskm_model__c_{n_clusters}.joblib"
    if not force_create and os.path.exists(filename):
        model = load(filename)
    else:
        print("No saved model found, clustering...")
        # Clustering DTW
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", n_jobs=-1, max_iter=10, random_state=42)
        model.fit_predict(X)
        dump(model, filename)
        print("Clustering done")
    return model

def get_distance_matrix(X):
    filename = "data/distance_matrix.joblib"
    if os.path.exists(filename):
        dist_matrix = load(filename)
    else:
        print(f"No saved distance matrix found, computing {X.shape[0] * (X.shape[0] + 1) // 2} tasks ...")
        dist_matrix = cdist_dtw(X, n_jobs=-1, verbose=1)
        dump(dist_matrix, filename)
    return dist_matrix

def main():
    # If true, run the cycle and fit with all cluster in [2,10]
    fit_all = False
    # Define feature to use
    features = ['close', 'd2c', 'shorts', 'volume', 'trend_score', 'news_volume', 'wiki_views']
    # Define number of cluster to visualize
    n_clusters = 3

    # Get data
    print("Reading data...")
    X_no_meme, ticker_no_meme = read_data(columns=features, file_name="dataset_no-meme.csv")
    X_meme, ticker_meme = read_data(columns=features, file_name="dataset_meme.csv", max_length=123)
    # Concatenate dataset
    X = np.vstack([X_meme, X_no_meme])
    ticker = np.hstack([ticker_meme, ticker_no_meme])
    # Feature normalization
    X = TimeSeriesScalerMeanVariance().fit_transform(X)

    # Fit models
    if fit_all:
        for c in range(2, 11):
            print(f"Fitting model with n_clusters={c}")
            get_cluster(X, c, force_create=True)


    # Analysis Silhouette score
    print("Starting Silhouette analysis...")
    # Compute distance matrix
    dist_matrix = get_distance_matrix(X)
    # Compute score
    k_values = np.arange(2, 11)
    scores = np.zeros(k_values.shape[0])
    for i, k in enumerate(k_values):
        model = get_cluster(X, k)
        labels = model.labels_
        # Silhouette with precomputed distance matrix
        score = silhouette_score(dist_matrix, labels, metric="precomputed")
        scores[i] = score
        print(f"Silhouette score with k={k}: {score:.3f}")
    # Plot silhouette score
    plt.figure(figsize=(7, 4))
    plt.plot(k_values, scores, marker='o', linestyle='-', linewidth=2)
    plt.axvline(x=3, color='r', linestyle='--', label='Optimal k = 3')
    plt.title('Silhouette Score vs Number of Clusters (DTW)', fontsize=12)
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette score')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


    # Analysis with fixed number of clusters
    print(f"Analysis with {n_clusters} clusters...")
    # Get model
    model = get_cluster(X, n_clusters)
    labels = model.labels_

    # Print cluster prediction
    for c in range(n_clusters):
        ticker_in_cluster = ticker[labels == c]
        print(f"Dimension of cluster {c+1}: {ticker_in_cluster.shape[0]}, "
              f"meme stock in cluster {c+1}: {sum([t in ticker_meme for t in ticker_in_cluster])}")

    # Visualization
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


if __name__ == '__main__':
    main()

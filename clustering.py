from joblib import dump, load
import matplotlib.pyplot as plt
import numpy as np
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.utils import to_time_series_dataset
import pandas as pd
import os


def read_data(columns, file_name="merged_data.csv"):
    # Read file
    file = pd.read_csv(f"data/{file_name}", usecols=["ticker"] + columns)
    # Group by ticker
    groups = file.groupby("ticker", sort=False)
    # Compute maximum length
    max_length = groups.size().max()
    # Define array
    X = np.zeros((len(groups), max_length, len(columns)), dtype=np.float32)
    # Fill array
    for i, (ticker, group) in enumerate(groups):
        vals = group[columns].values
        X[i, :len(vals), :] = vals
    # Return dataset
    return to_time_series_dataset(X), np.array(list(groups.groups.keys()))

def get_cluster(X, n_clusters):
    filename = f"tskm_model__c_{n_clusters}.joblib"
    if os.path.exists(filename):
        model = load(filename)
    else:
        print("No saved model found, clustering...")
        # Clustering DTW
        model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", n_jobs=-1, max_iter=10, random_state=42)
        model.fit_predict(X)
        dump(model, filename)
        print("Clustering done")
    return model

def main():
    # Define feature to use
    features = ['d2c', 'shorts', 'volume', 'close', 'news_volume']
    #features = ['d2c', 'shorts', 'volume', 'close', 'trend']
    # Define number of cluster
    n_clusters = 3

    # Get data
    print("Reading data...")
    X, ticker = read_data(columns=features, file_name="merged_data.csv")
    # Feature normalization
    X = TimeSeriesScalerMeanVariance().fit_transform(X)

    model = get_cluster(X, n_clusters)
    labels = model.labels_

    # Print cluster prediction
    for c in range(n_clusters):
        print(f"Ticker in cluster {c}:", ticker[labels == c])

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

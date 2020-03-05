import pandas as pd
import numpy as np


def get_random_centroids(df, k):
    centroids = []
    for i in range(0, k):
        centroids.append(np.random.uniform(min(df.min()), max(df.max()), df.shape[1]))
    return np.array(centroids)


def recalculate_centroids():
    return None


def get_cluster_labels(df, centroids):
    labels = np.zeros(df.shape[0])
    for row in df.to_numpy():
        np.linalg.norm(row - )
    return labels


def stop_condition(old_centroids, centroids, iterations):
    if iterations > 100:
        return True
    return old_centroids == centroids


def k_means(df, k):
    iterations = 0
    centroids = get_random_centroids(df, k)
    old_centroids = np.zeros(centroids.shape)

    while not stop_condition(old_centroids, centroids, iterations):
        iterations += 1
        old_centroids = centroids

        cluster_labels = get_cluster_labels(df, centroids)

        centroids = recalculate_centroids()

    return None


# Read heart disease data from CSV
heart_disease_df = pd.read_csv("data.csv")

# Min-max normalize original dataset to prepare for clustering
normalized_df = (heart_disease_df - heart_disease_df.min()) / (heart_disease_df.max() - heart_disease_df.min())

# k_means(normalized_df, 3)

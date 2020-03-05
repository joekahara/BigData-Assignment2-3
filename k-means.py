import pandas as pd
import numpy as np
from copy import deepcopy


def get_random_centroids(data, k):
    centroids = []
    for _ in range(k):
        centroids.append(np.random.uniform(data.min(), data.max(), data.shape[1]))
    return np.array(centroids)


def get_cluster_distances(data, centroids):
    distances = []
    for row in data:
        row_distances = []
        for c in centroids:
            row_distances.append(np.linalg.norm(row - c))
        distances.append(row_distances)
    return np.array(distances)


def get_cluster_labels(distances):
    labels = []
    for row in distances:
        labels.append(np.where(row == np.amin(row))[0])
    return np.array(labels)


def recalculate_centroids(data, labels, k):
    new_centroids = []
    for i in range(k):
        datapoints = np.array([data[j] for j in range(len(data)) if labels[j] == i])
        new_centroids.append(np.mean(datapoints))
    return np.array(new_centroids)


def k_means(df, k):
    iterations = 0

    data = np.array(list(df.values))

    centroids = get_random_centroids(data, k)
    old_centroids = np.zeros(centroids.shape)

    while iterations < 100 and not (centroids.all() == old_centroids.all()):
        iterations += 1

        distances = get_cluster_distances(data, centroids)
        cluster_labels = get_cluster_labels(distances)

        old_centroids = deepcopy(centroids)
        centroids = recalculate_centroids(data, cluster_labels, k)

    return np.reshape(cluster_labels, len(cluster_labels))


def silhouette_width(df, k):
    point_distances = []
    data = np.array(list(df.values))
    print(data)
    '''for i in range(k):
        if point_distances.append = np.array([data[j] for j in range(len(data)) if labels[j] == i])
        new_centroids.append(np.mean(datapoints))
    return None'''


# Read heart disease data from CSV
heart_disease_df = pd.read_csv("data.csv")

# Min-max normalize original dataset to prepare for clustering
normalized_df = (heart_disease_df - heart_disease_df.min()) / (heart_disease_df.max() - heart_disease_df.min())

clusters_3 = normalized_df.assign(cluster=pd.Series(k_means(normalized_df, 3)).values)
print(clusters_3)
print("Silhouette width for k = 3: ", silhouette_width(clusters_3, 3), '\n')

clusters_6 = normalized_df.assign(cluster=pd.Series(k_means(normalized_df, 6)).values)
print(clusters_6)
print("Silhouette width for k = 6: ", silhouette_width(clusters_6, 6), '\n')

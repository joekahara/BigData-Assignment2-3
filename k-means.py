import pandas as pd
import numpy as np
from scipy.spatial.distance import cdist
from copy import deepcopy
from sklearn.datasets import make_blobs


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
        cluster_points = np.array([data[j] for j in range(len(data)) if labels[j] == i])
        new_centroids.append(np.mean(cluster_points))
    return np.array(new_centroids)


def k_means(df, k):
    iterations = 0
    data = df.to_numpy()
    centroids = get_random_centroids(data, k)
    old_centroids = np.zeros(centroids.shape)
    cluster_labels = None
    while iterations < 100 and not (centroids.all() == old_centroids.all()):
        iterations += 1
        distances = cdist(data, centroids)
        cluster_labels = get_cluster_labels(distances)
        old_centroids = deepcopy(centroids)
        centroids = recalculate_centroids(data, cluster_labels, k)
    print("Average silhouette width for k =", k, ": ", average_silhouette_width(data, cluster_labels))
    return np.reshape(cluster_labels, len(cluster_labels))


def average_silhouette_width(data, labels):
    # Initialize list to store all points' silhouette widths
    silhouette_widths = []

    # Combine data and cluster labels into one np array
    labeled_data = np.append(data, labels, axis=1)

    # Sort data by cluster label, effectively grouping them
    sorted_data = labeled_data[labeled_data[:, -1].argsort()]

    # Split into nested numpy arrays for each different cluster label
    # We now have an array of arrays of data within each separate cluster
    data_by_cluster = np.split(sorted_data, np.where(np.diff(sorted_data[:, -1]))[0] + 1)

    # Remove cluster labels from np array, no longer needed
    for i in range(len(data_by_cluster)):
        data_by_cluster[i] = np.delete(data_by_cluster[i], -1, 1)

    for i in range(len(data_by_cluster)):
        for point in data_by_cluster[i]:
            # print(point)
            # Find value of a for each point
            a = np.mean(cdist(point.reshape(1, -1), data_by_cluster[i]))
            distance_from_clusters = []

            # Find value of b for each point
            for j in range(len(data_by_cluster)):
                if not i == j:
                    distance_from_clusters.append(np.mean(cdist(point.reshape(1, -1), data_by_cluster[j])))
            b = np.min(distance_from_clusters)

            # Append all points' silhouette width to list
            silhouette_widths.append((b-a)/max(a, b))

    # Return mean of list for average silhouette width
    return np.mean(silhouette_widths)


# Read heart disease data from CSV
heart_disease_df = pd.read_csv("k-means-data.csv")

# Min-max normalize original dataset to prepare for clustering
normalized_df = (heart_disease_df - heart_disease_df.min()) / (heart_disease_df.max() - heart_disease_df.min())

clusters_3 = normalized_df.assign(cluster=pd.Series(k_means(normalized_df, 3)).values)
clusters_3.to_csv("clusters_3.csv")
print(clusters_3, '\n')

clusters_6 = normalized_df.assign(cluster=pd.Series(k_means(normalized_df, 6)).values)
clusters_6.to_csv("clusters_6.csv")
print(clusters_6, '\n')

# Use sklearn to generate dataset w 3 centers to test silhouette width
X, y = make_blobs(n_samples=100, centers=3, n_features=5)
print("Testing with manually generated dataset with 3 centers: ")
k_means(pd.DataFrame(X), 3)

#exp 8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

iris = datasets.load_iris()
X = iris.data  # features (sepal length, sepal width, petal length, petal width)
y = iris.target  # actual labels (not used for clustering, just for checking accuracy)


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)  # 3 clusters (Iris-setosa, versicolor, virginica)
kmeans.fit(X_scaled)


labels = kmeans.labels_
centroids = kmeans.cluster_centers_


print("Cluster labels assigned:\n", labels)
print("Actual labels:\n", y)


plt.figure(figsize=(8, 6))
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels, cmap='viridis', marker='o', edgecolor='k', s=80)
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='X', s=200, label='Centroids')
plt.title("K-Means Clustering on Iris Dataset")
plt.xlabel("Feature 1 (Standardized Sepal Length)")
plt.ylabel("Feature 2 (Standardized Sepal Width)")
plt.legend()
plt.show()
#exp 10

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load Iris dataset
iris = load_iris()
X = iris.data  # shape (150,4)
y = iris.target

#  Normalize features (optional but recommended)
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

#  Create similarity matrix (cosine similarity)
similarity_matrix = cosine_similarity(X_scaled)
np.fill_diagonal(similarity_matrix, 0)  # remove self-links

#  Convert similarity matrix to stochastic matrix (column sums = 1)
column_sums = similarity_matrix.sum(axis=0)
stochastic_matrix = similarity_matrix / column_sums

# Initialize PageRank vector
n = X.shape[0]  # number of data points
pr = np.ones(n) / n  # initial equal ranks


#  PageRank parameters
d = 0.85  # damping factor
max_iter = 100
tol = 1e-6

#  Iterative PageRank calculation
for i in range(max_iter):
    pr_new = (1 - d) / n + d * stochastic_matrix.dot(pr)
    if np.linalg.norm(pr_new - pr, 1) < tol:
        break
    pr = pr_new

#  Create DataFrame to show results
df_pr = pd.DataFrame({
    'Index': range(n),
    'PageRank': pr,
    'Class': y
}).sort_values(by='PageRank', ascending=False)

print(df_pr.head(10))
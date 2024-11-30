"""
Deep Learning on Graphs - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye

from sklearn.linear_model import LogisticRegression
from sklearn.manifold import SpectralEmbedding
from sklearn.metrics import accuracy_score
from deepwalk import deepwalk


# Loads the karate network
G = nx.read_weighted_edgelist('../data/karate.edgelist', delimiter=' ', nodetype=int, create_using=nx.Graph())
print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())

n = G.number_of_nodes()

# Loads the class labels
class_labels = np.loadtxt('../data/karate_labels.txt', delimiter=',', dtype=np.int32)
idx_to_class_label = dict()
for i in range(class_labels.shape[0]):
    idx_to_class_label[class_labels[i, 0]] = class_labels[i, 1]

y = list()
for node in G.nodes():
    y.append(idx_to_class_label[node])

y = np.array(y)

############## Task 5
# Visualizes the karate network
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
nx.draw_networkx(G, with_labels=True, label=y, node_color=y, cmap=plt.cm.tab20, node_size=300)
plt.show()

############## Task 6
# Extracts a set of random walks from the karate network and feeds
# them to the Skipgram model
n_dim = 128
n_walks = 10
walk_length = 20
model = deepwalk(G, n_walks, walk_length, n_dim)

embeddings = np.zeros((n, n_dim))
for i, node in enumerate(G.nodes()):
    embeddings[i, :] = model.wv[str(node)]

idx = np.random.RandomState(seed=42).permutation(n)
idx_train = idx[:int(0.8*n)]
idx_test = idx[int(0.8*n):]

X_train = embeddings[idx_train, :]
X_test = embeddings[idx_test, :]

y_train = y[idx_train]
y_test = y[idx_test]


############## Task 7
# Trains a logistic regression classifier and use it to make predictions

cls_dw = LogisticRegression()
cls_dw.fit(X_train, y_train)
y_pred = cls_dw.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy with deepwalk: ", accuracy)

############## Task 8
# Generates spectral embeddings

adjacency = nx.adjacency_matrix(G)
degree_matrix = np.array([d for _, d in G.degree()])
degree_matrix = diags(degree_matrix)

laplacian = degree_matrix - adjacency

degree_inv_sqrt = diags(1.0 / np.sqrt(degree_matrix.diagonal()))
laplacian_normalized = degree_inv_sqrt @ laplacian @ degree_inv_sqrt

n_dim = 2
_, U = eigs(laplacian_normalized, k=n_dim, which='SM')

U = U.real
# show our data with the spectral embedding
plt.figure(figsize=(10, 10))
plt.scatter(U[:, 0], U[:, 1], c=y, cmap=plt.cm.tab20)
plt.show()

# comparaison with DeepWalk in terms of accuracy

X_train_s = U[idx_train, :]
X_test_s = U[idx_test, :]

cls_s = LogisticRegression()
cls_s.fit(X_train_s, y_train)
y_pred_s = cls_s.predict(X_test_s)

accuracy_s = accuracy_score(y_test, y_pred_s)
print("Accuracy with spectral embedding: ", accuracy_s)
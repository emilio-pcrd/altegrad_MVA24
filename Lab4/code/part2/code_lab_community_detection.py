"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans



############## Task 3
# Perform spectral clustering to partition graph G into k clusters
def spectral_clustering(G, k):

    ##################
    # your code here #
    ##################
    adjency_matrix = nx.adjacency_matrix(G)
    diag_degree_matrix = diags(np.array(adjency_matrix.sum(axis=1)).flatten())
    right_component = np.linalg.inv(diag_degree_matrix) @ adjency_matrix
    Laplacian_matrix = eye(adjency_matrix.shape[0]) - right_component

    eigenvalues, eigenvectors = eigs(Laplacian_matrix, k=k, which='SM')
    U = eigenvectors[:, :k]
    
    kmeans = KMeans(n_clusters=k).fit(U)
    labels = kmeans.labels_

    # dict node_id -> cluster_id
    clustering = {node: labels[i] for i, node in enumerate(G.nodes())}

    return clustering


############## Task 4

##################
# your code here #
##################




############## Task 5
# Compute modularity value from graph G based on clustering
def modularity(G, clustering):
    
    ##################
    # your code here #
    ##################
    
    
    
    
    return modularity



############## Task 6

##################
# your code here #
##################








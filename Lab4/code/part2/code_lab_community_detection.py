"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.sparse import diags, eye
from random import randint
from sklearn.cluster import KMeans


##############
# Task 3

def spectral_clustering(G, k):
    adjency_matrix = nx.adjacency_matrix(G)
    diag_degree_matrix = diags(np.array(1/adjency_matrix.sum(axis=1)).flatten())
    right_component = diag_degree_matrix @ adjency_matrix
    Laplacian_matrix = eye(adjency_matrix.shape[0]) - right_component

    _, eigenvectors = eigs(Laplacian_matrix, k=k, which='SM')
    eigenvectors = np.real(eigenvectors)
    U = eigenvectors[:, :k]

    kmeans = KMeans(n_clusters=k).fit(U)
    labels = kmeans.labels_

    # dict node_id -> cluster_id
    clustering = {node: labels[i] for i, node in enumerate(G.nodes())}

    return clustering


##############
# Task 4

##################
# your code here #
##################
__path__ = "../datasets/CA-HepTh.txt"
G = nx.read_edgelist(__path__, delimiter='\t')

connected_components = list(nx.connected_components(G))
largest_component = max(connected_components, key=len)
largest_component_subgraph = G.subgraph(largest_component)

clustering = spectral_clustering(largest_component_subgraph, k=50)


##############
# Task 5

def modularity(G, clustering):

    ##################
    # your code here #
    ##################
    # total number of edges
    m = G.number_of_edges()

    # number of communities
    n_c = len(np.unique(list(clustering.values())))

    # number of edges in each community
    l_c = np.zeros(n_c)
    for n1, n2 in G.edges():
        if clustering[n1] == clustering[n2]:
            l_c[clustering[n1]] += 1

    # sum of the degrees of the nodes in each community
    d_c = np.zeros(n_c)
    for node in G.nodes():
        d_c[clustering[node]] += G.degree(node)

    # modularity
    modularity = np.sum(l_c / m - (d_c / (2*m))**2)
    return modularity


##############
#  Task 6

#  clustering obtained by spectral clustering with k=50
modularity_val = modularity(
    largest_component_subgraph,
    clustering
    )

print(f"Modularity value: {modularity_val}")


# clustering obtained with random partition
# of the nodes into 50 clusters

clustering_random = {node: randint(0, 49) for node in largest_component_subgraph.nodes()}
modularity_val_random = modularity(
    largest_component_subgraph,
    clustering_random
    )

print(f"Modularity value with random partition: {modularity_val_random}")

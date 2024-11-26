"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx


############## Task 7

# load Mutag dataset
def load_dataset():

    dataset = TUDataset(root='../dataset/MUTAG/', name='MUTAG')
    Gs = [to_networkx(data).to_undirected() for data in dataset]
    y = [data.y.item() for data in dataset]
    return Gs, y


Gs, y = load_dataset()
G_train, G_test, y_train, y_test = train_test_split(Gs, y, test_size=0.2, random_state=42)

# Compute the shortest path kernel
def shortest_path_kernel(Gs_train, Gs_test):
    all_paths = dict()
    sp_counts_train = dict()

    for i, G in enumerate(Gs_train):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_train[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_train[i]:
                        sp_counts_train[i][length] += 1
                    else:
                        sp_counts_train[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    sp_counts_test = dict()

    for i, G in enumerate(Gs_test):
        sp_lengths = dict(nx.shortest_path_length(G))
        sp_counts_test[i] = dict()
        nodes = G.nodes()
        for v1 in nodes:
            for v2 in nodes:
                if v2 in sp_lengths[v1]:
                    length = sp_lengths[v1][v2]
                    if length in sp_counts_test[i]:
                        sp_counts_test[i][length] += 1
                    else:
                        sp_counts_test[i][length] = 1

                    if length not in all_paths:
                        all_paths[length] = len(all_paths)

    phi_train = np.zeros((len(Gs_train), len(all_paths)))
    for i in range(len(Gs_train)):
        for length in sp_counts_train[i]:
            phi_train[i, all_paths[length]] = sp_counts_train[i][length]

    phi_test = np.zeros((len(Gs_test), len(all_paths)))
    for i in range(len(Gs_test)):
        for length in sp_counts_test[i]:
            phi_test[i, all_paths[length]] = sp_counts_test[i][length]

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


############## Task 8
# Compute the graphlet kernel
def graphlet_kernel(Gs_train, Gs_test, n_samples=200):
    graphlets = [nx.Graph(), nx.Graph(), nx.Graph(), nx.Graph()]
    # create all graphlets of size 3
    graphlets[0].add_nodes_from(range(3))

    graphlets[1].add_nodes_from(range(3))
    graphlets[1].add_edge(0, 1)

    graphlets[2].add_nodes_from(range(3))
    graphlets[2].add_edge(0, 1)
    graphlets[2].add_edge(1, 2)

    graphlets[3].add_nodes_from(range(3))
    graphlets[3].add_edge(0, 1)
    graphlets[3].add_edge(1, 2)
    graphlets[3].add_edge(0, 2)

    phase = ['train', 'test']

    phi_train = np.zeros((len(G_train), 4))
    phi_test = np.zeros((len(G_test), 4))

    for phase in ['train', 'test']:
        if phase == 'train':
            Graph = G_train
            phi = phi_train
        else:
            Graph = G_test
            phi = phi_test

        for i, G in enumerate(Graph):
            for _ in range(n_samples):
                s = np.random.choice(G.nodes(), 3)
                subgraph = G.subgraph(s)

                # check if subgraph is isomorphic to one of the graphlets
                for j, graphlet in enumerate(graphlets):
                    if nx.is_isomorphic(subgraph, graphlet):
                        phi[i, j] += 1
                        break

    K_train = np.dot(phi_train, phi_train.T)
    K_test = np.dot(phi_test, phi_train.T)

    return K_train, K_test


K_train_sp, K_test_sp = shortest_path_kernel(G_train, G_test)

############## Task 9

K_train_gk, K_test_gk = graphlet_kernel(G_train, G_test)


############## Task 10

from sklearn.svm import SVC

# FOR SHORTEST PATH KERNEL
# initialize SVM and train
clf_sp = SVC(kernel='precomputed')
clf_sp.fit(K_train_sp, y_train)

# predict
y_pred_sp = clf_sp.predict(K_test_sp)
sp_acc_score = accuracy_score(y_test, y_pred_sp)
print("Accuracy score for shortest path kernel: ", sp_acc_score)

# FOR SHORTEST PATH KERNEL
# initialize SVM and train
clf_gk = SVC(kernel='precomputed')
clf_gk.fit(K_train_gk, y_train)

# predict
y_pred_gk = clf_sp.predict(K_test_gk)
gk_acc_score = accuracy_score(y_test, y_pred_gk)
print("Accuracy score for graphlet kernel: ", gk_acc_score)

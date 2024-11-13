"""
Graph Mining - ALTEGRAD - Nov 2024
"""

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


############## Task 1
__path__ = '../datasets/CA-HepTh.txt'
G = nx.read_edgelist(__path__, delimiter='\t')
tot_nodes_G = G.number_of_nodes()
tot_edges_G = G.number_of_edges()
print(f"nuber of edge in G: {tot_edges_G}")
print(f"nuber of nodes in G: {tot_nodes_G} \n")


############## Task 2
connected_components = list(nx.connected_components(G))
print(f"number of connected components in G: {len(connected_components)}")

largest_component = max(connected_components, key=len)
largest_component_subgraph = G.subgraph(largest_component)

num_nodes_largest_component = largest_component_subgraph.number_of_nodes()
num_edges_largest_component = largest_component_subgraph.number_of_edges()
print(f"number of nodes in the largest connected component: {num_nodes_largest_component}")
print(f"number of edges in the largest connected component: {num_edges_largest_component} \n")

fraction_nodes = num_nodes_largest_component / tot_nodes_G
fraction_edges = num_edges_largest_component / tot_edges_G
print(f"fraction of nodes in the largest connected component: {fraction_nodes}")
print(f"fraction of edges in the largest connected component: {fraction_edges} \n")

import numpy as np
import networkx as nx
from itertools import product


def generate_random_graph(n_nodes, edge_prob=0.5, seed=None):
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    return G


def generate_regular_graph(n_nodes, degree=3, seed=None):
    if (n_nodes * degree) % 2 != 0:
        raise ValueError("n_nodes * degree must be even")
    G = nx.random_regular_graph(degree, n_nodes, seed=seed)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    return G


def generate_cycle_graph(n_nodes):
    G = nx.cycle_graph(n_nodes)
    for u, v in G.edges():
        G[u][v]['weight'] = 1.0
    return G


def brute_force_maxcut(G):
    # obviously only works for small graphs lol
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    best_cut = 0
    best_partition = None

    for bits in product([0, 1], repeat=n):
        cut_value = 0
        for u, v, data in G.edges(data=True):
            i = nodes.index(u)
            j = nodes.index(v)
            if bits[i] != bits[j]:
                cut_value += data.get('weight', 1.0)

        if cut_value > best_cut:
            best_cut = cut_value
            best_partition = bits

    return best_cut, best_partition


def compute_cut_value(G, partition):
    nodes = list(G.nodes())
    cut_val = 0
    for u, v, data in G.edges(data=True):
        i = nodes.index(u)
        j = nodes.index(v)
        if partition[i] != partition[j]:
            cut_val += data.get('weight', 1.0)
    return cut_val


def get_maxcut_operator(G):
    n = G.number_of_nodes()
    nodes = list(G.nodes())

    pauli_terms = []
    offset = 0.0

    for u, v, data in G.edges(data=True):
        w = data.get('weight', 1.0)
        i = nodes.index(u)
        j = nodes.index(v)

        offset += 0.5 * w

        pauli_str = ['I'] * n
        pauli_str[i] = 'Z'
        pauli_str[j] = 'Z'
        pauli_terms.append((-0.5 * w, ''.join(pauli_str)))

    return pauli_terms, offset
# TODO: add support for weighted graphs with different weights

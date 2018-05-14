import itertools
from graphtools import *
import pandas as pd

def internal_interconnectivity(graph, cluster):
    return np.sum(bisection_weights(graph, cluster))


def relative_interconnectivity(graph, cluster_i, cluster_j):
    edges = connecting_edges((cluster_i, cluster_j), graph)
    EC = np.sum(get_weights(graph, edges))
    ECci, ECcj = internal_interconnectivity(graph, cluster_i), internal_interconnectivity(graph, cluster_j)
    return EC / ((ECci + ECcj) / 2.0)


def internal_closeness(graph, cluster):
    cluster = graph.subgraph(cluster)
    edges = cluster.edges()
    weights = get_weights(cluster, edges)
    return np.sum(weights)


def relative_closeness(graph, cluster_i, cluster_j):
    edges = connecting_edges((cluster_i, cluster_j), graph)
    if not edges:
        SEC = 1e4
    else:
        SEC = np.mean(get_weights(graph, edges))
    Ci, Cj = internal_closeness(graph, cluster_i), internal_closeness(graph, cluster_j)
    SECci, SECcj = np.mean(bisection_weights(graph, cluster_i)), np.mean(bisection_weights(graph, cluster_j))
    return SEC / ((Ci / (Ci + Cj) * SECci) + (Cj / (Ci + Cj) * SECcj))


merge_score = lambda g, ci, cj, a: relative_interconnectivity(g, ci, cj) * np.power(relative_closeness(g, ci, cj), a)


def merge_best(graph, df, a, stop_condition, verbose=False, threshold=0):
    clusters = np.unique(df['cluster'])
    if len(clusters) <= stop_condition:
        return False
    max_score = 0
    ci, cj = -1, -1

    for combination in itertools.combinations(clusters, 2):
        i, j = combination
        if i != j:
            if verbose: print("Checking c%d c%d" % (i, j))
            ms = merge_score(graph, get_cluster(graph, [i]), get_cluster(graph, [j]), a)
            if verbose: print("Merge score: %f" % (ms))
            if ms > max_score:
                if verbose: print("Better than: %f" % (max_score))
                max_score = ms
                ci, cj = i, j

    if max_score > threshold:
        if verbose: print("Merging c%d and c%d" % (ci, cj))
        df.loc[df['cluster'] == cj, 'cluster'] = ci
    return max_score > threshold


def cluster_encodings(encodings, children=2):
    paths = []
    values = []
    for key, value in encodings.items():
        paths.append(key)
        values.append(value)
    df = pd.DataFrame([v.transpose() for v in values])
    graph = knn_graph(df, 6, verbose=True)
    graph = part_graph(graph, children, df)
    # while merge_best(graph, df, 1, children, verbose=False):
    #     pass

    clusters = {}
    for idx, cluster in enumerate(df['cluster']):
        if cluster not in clusters:
            clusters[cluster] = []
        clusters[cluster].append(paths[idx])

    return sorted(clusters.values(), key=len, reverse=True)
import networkx as nx
from typing import List, Tuple, Dict
import numpy.typing as npt
import numpy as np
import utils

GRAPH_SCALING_RESOLUTION = 100_000


def _to_nx_graph(cluster_scores: List[npt.NDArray], fix_weighting=True, weighting_per_edge=None, corr_cutoff=None):
    node_idx = 0
    assert weighting_per_edge is None or len(weighting_per_edge) == len(
        cluster_scores), "Need a weight for each edge from prior layer to next"
    # We need to account for all outgoing from the end
    n_clusters = sum([len(cs) for cs in cluster_scores]) + \
        len(cluster_scores[-1][0])
    most_pos_per_layer = [cs.max() * (1 if weighting_per_edge is None else weighting_per_edge[i])
                          for i, cs in enumerate(cluster_scores)]  # + eps

    G = nx.DiGraph()
    graph_idx_to_node_idx = [{}]
    node_idx_to_graph_idx = [{}]
    for layer in range(len(cluster_scores)):
        # Append the next layer
        graph_idx_to_node_idx.append({})
        node_idx_to_graph_idx.append({})
        layer_start_idx = node_idx
        n_in_layer = len(cluster_scores[layer])
        for i, node_cs in enumerate(cluster_scores[layer]):
            graph_idx_to_node_idx[layer][node_idx] = i
            node_idx_to_graph_idx[layer][i] = node_idx
            for j, c in enumerate(node_cs):
                next_idx = layer_start_idx + n_in_layer + j

                if fix_weighting:
                    # We need all the weights to be positive but the shortest paths to be the most important
                    w = round((-1 * c *
                               (1 if weighting_per_edge is None else weighting_per_edge[layer])
                               + most_pos_per_layer[layer])
                              * GRAPH_SCALING_RESOLUTION)
                    assert w >= 0, f"Weight is negative: {w}"
                else:
                    w = c

                graph_idx_to_node_idx[layer + 1][next_idx] = j
                node_idx_to_graph_idx[layer + 1][j] = next_idx
                if w > 0 and (corr_cutoff is None or c > corr_cutoff):
                    G.add_edge(node_idx, next_idx, weight=w)
            node_idx += 1

    sink = n_clusters
    source = n_clusters + 1

    for i in range(len(cluster_scores[0])):
        G.add_edge(source, node_idx_to_graph_idx[0][i], weight=1)
    for i in range(len(cluster_scores[-1][0])):
        G.add_edge(node_idx_to_graph_idx[-1][i], sink, weight=1)

    # nx.draw(G, with_labels=True, pos=nx.nx_pydot.graphviz_layout(G, prog='dot'))
    # plt.savefig("graph.png")
    return G, source, sink, graph_idx_to_node_idx, node_idx_to_graph_idx, most_pos_per_layer


def top_k_dag_paths(layers: List[npt.NDArray], layer: int, neuron: int, k: int,
                    weighting_per_edge: List[float] = None, corr_cutoff = 0.01, exclude_set={}):
    if weighting_per_edge is None:
        weighting_per_edge = [1.0 for _ in layers]
    # TODO: we no longer have to remake these graphs. One and done
    r = utils.restrict_to_related_vertex(layers, layer, neuron)
    graph, source, sink, graph_layers_to_idx, \
        node_layers_to_graph, most_pos_per_layer = _to_nx_graph(
            r, weighting_per_edge=weighting_per_edge, corr_cutoff=corr_cutoff)

    for rm_layer in exclude_set.keys():
        for node in exclude_set[rm_layer]:
            print("Removing", rm_layer, node,
                  node_layers_to_graph[rm_layer][node])
            graph.remove_node(node_layers_to_graph[rm_layer][node])

    node_start  = node_layers_to_graph[layer][0] # Because we select for only 1
    X = nx.shortest_simple_paths(graph, node_start, sink, weight='weight')

    paths = []
    for counter, path in enumerate(X):
        path_no_sink = path[:-1]
        # print(path_no_sink, path)
        #  TODO: CANNOT GO BACKWARDS
        # print(path, path_no_sink_no_source)
        # print("PATH NO SINK", path_no_sink)
        path_node_idx = [graph_layers_to_idx[i + layer][node]
                         for i, node in enumerate(path_no_sink)]
        assert len(path_node_idx) == len(layers) + 1 - layer
        path_node_idx[0] = neuron

        recovered_weight = sum([
            -1 * weighting_per_edge[i] * (graph[path_no_sink[i]][path_no_sink[i + 1]]['weight'] / GRAPH_SCALING_RESOLUTION \
                - most_pos_per_layer[i]) 
            for i in range(0, len(path_no_sink) - 1)])  
        
        paths.append((path_node_idx, recovered_weight))
        print(paths[-1])
        if counter == k-1:
            break
    return paths


def get_feature_paths(lattice, layer: int, neuron: int, k_search=20,
                      weighting_per_edge: List[float] = None, n_max_features=5):
    print(f"Getting top {k_search} paths")
    assert weighting_per_edge is None or len(
        weighting_per_edge) == len(lattice)
    assert len(lattice) > 1, "Need at least 2 layers"
    searched_paths = top_k_dag_paths(
        lattice, layer=layer, neuron=neuron, k=k_search,
        weighting_per_edge=weighting_per_edge)
    print(f"Got top {k_search} paths")
    paths = [p[0] for p in searched_paths]

    n_layers_in_path = len(lattice) - layer

    def cluster_similar_paths():
        # Get a pairwise "similarity" between the paths
        sims = np.zeros((k_search, k_search))
        for i in range(k_search):
            for j in range(k_search):
                for l in range(n_layers_in_path):
                    if paths[i][l] == paths[j][l]:
                        sims[i, j] += 1
        # Now we can employ a greedy-type algorithm
        clusters = []
        path_idx_to_cluster = np.zeros(k_search) - 1
        # A list of whether a path has been clustered
        clustered = np.zeros(k_search) == 1

        diff_cutoff = n_layers_in_path - 1

        def start_over():
            nonlocal curr_idx, curr_cluster
            curr_idx = 0
            curr_cluster = 0

            clusters.clear()
            path_idx_to_cluster.fill(-1)
            clustered.fill(False)

            clusters.append([curr_idx])
            path_idx_to_cluster[curr_idx] = curr_cluster
            clustered[curr_idx] = True

        start_over()

        while len(clusters) < n_max_features and diff_cutoff > 0:
            found_next = False
            for i in range(k_search):
                diff_from_all = True
                for c in clusters:
                    rep = c[0]
                    if not clustered[i] and sims[rep, i] <= n_layers_in_path - diff_cutoff:
                        pass
                    else:
                        diff_from_all = False
                if diff_from_all:
                    found_next = True
                    curr_idx = i
                    curr_cluster += 1
                    clusters.append([curr_idx])
                    clustered[curr_idx] = True
                    path_idx_to_cluster[curr_idx] = curr_cluster
                    break
            if not found_next:
                # start_over()
                diff_cutoff -= 1
                print("Trying with cluster diffence cutoff", diff_cutoff)
        return clusters

    paths_distinct = cluster_similar_paths()
    path_reps = [searched_paths[p[0]] for p in paths_distinct]
    return path_reps

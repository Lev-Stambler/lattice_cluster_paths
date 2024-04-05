import networkx as nx
from typing import List, Tuple, Dict
import numpy.typing as npt
import numpy as np
import utils

GRAPH_SCALING_RESOLUTION = 100_000

def to_nx_graph(cluster_scores: List[npt.NDArray]) -> Tuple[nx.DiGraph, int, List[Dict[int, int]], List[Dict[int, int]]]:
    node_idx = 0
    # We need to account for all outgoing from the end
    n_clusters = sum([len(cs) for cs in cluster_scores]) + \
        len(cluster_scores[-1][0])
    most_pos = (max([cs.max() for cs in cluster_scores]))  # + eps

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
                    
                # We need all the weights to be positive but the shortest paths to be the most important
                w = round((-1 * c + most_pos)
                          * GRAPH_SCALING_RESOLUTION)
                assert w >= 0, f"Weight is negative: {w}"

                graph_idx_to_node_idx[layer + 1][next_idx] = j
                node_idx_to_graph_idx[layer + 1][j] = next_idx
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
    return G, source, sink, graph_idx_to_node_idx, node_idx_to_graph_idx, most_pos

def top_k_dag_paths(layers: List[npt.NDArray], layer: int, neuron: int, k: int, exclude_set = {}):
    r = utils.restrict_to_related_vertex(layers, layer, neuron)
    # print(r)
    graph, source, sink, graph_layers_to_idx, node_layers_to_graph, most_pos = to_nx_graph(
        r)
    
    for rm_layer in exclude_set.keys():
        for node in exclude_set[rm_layer]:
            print("Removing", rm_layer, node, node_layers_to_graph[rm_layer][node])
            graph.remove_node(node_layers_to_graph[rm_layer][node])

    X = nx.shortest_simple_paths(graph, source, sink, weight='weight')
    
    paths = []
    for counter, path in enumerate(X):
        path_no_sink_no_source = path[1:-1]
        # print(path_no_sink, path)
        #  TODO: CANNOT GO BACKWARDS
        # print(path, path_no_sink_no_source)
        path_node_idx = [graph_layers_to_idx[i][node]
                         for i, node in enumerate(path_no_sink_no_source)]
        assert len(path_node_idx) == len(layers) + 1
        path_node_idx[layer] = neuron
        total_weight = sum([graph[path[i]][path[i + 1]]['weight']
                            for i in range(len(path) - 1)])
        total_weight_no_sink = total_weight - 1
        recovered_weight = -1 * (total_weight_no_sink / GRAPH_SCALING_RESOLUTION - most_pos * len(path_no_sink_no_source))
        paths.append((path_node_idx, recovered_weight))
        # print(paths[-1])
        if counter == k-1:
            break
    return paths

def get_feature_paths(lattice, layer: int, neuron: int, k_search=20, n_max_features=5):
    print(f"Getting top {k_search} paths")
    searched_paths = utils.top_k_dag_paths(lattice, layer=layer, neuron=neuron, k=k_search)
    print(f"Got top {k_search} paths")
    paths = [p[0] for p in searched_paths]

    n_layers = len(lattice)
    def cluster_similar_paths():
        # Get a pairwise "similarity" between the paths
        sims = np.zeros((k_search, k_search))
        for i in range(k_search):
            for j in range(k_search):
                for l in range(n_layers):
                    if paths[i][l] == paths[j][l]:
                        sims[i, j] += 1
        # Now we can employ a greedy-type algorithm
        clusters = []
        path_idx_to_cluster = np.zeros(k_search) - 1
        clustered = np.zeros(k_search) == 1 # A list of whether a path has been clustered

        diff_cutoff = n_layers - 1
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
                    if not clustered[i] and sims[rep, i] <= n_layers - diff_cutoff:
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

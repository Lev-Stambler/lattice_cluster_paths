from typing import List, Dict, Tuple
import torch
import numpy as np
from einops import rearrange
import networkx as nx
import heapq
import numpy.typing as npt
# TODO: this is just a c and p rn

# Get the activations for the best dict features
# TODO: fix up
GRAPH_SCALING_RESOLUTION = 100_000


def top_k_dag_paths_dynamic(layers: List[List[List[float]]], k: int, top_layer: int = None):
    n_to_top_layer = len(layers[-1][0])
    layers = layers + [
        [[0] for _ in range(n_to_top_layer)]
    ]

    if top_layer is None:
        top_layer = len(layers)
    assert len(layers) > 1, "Need at least 2 layers"
    assert top_layer <= len(
        layers), "Top layer must be less than the number of layers"

    def recur(layer: int, node_layer_idx: int, memoizer):
        if (layer, node_layer_idx) in memoizer:
            return memoizer[(layer, node_layer_idx)]

        # Base case
        if layer == 0:
            # TODO: RETURN SOMETHING HERE
            return [([node_layer_idx], 0)]

        past_layer = layer - 1
        past_layer_vals = layers[past_layer]

        # Get the inbound values
        vs = [s[node_layer_idx] for s in past_layer_vals]

        paths = []
        for i, v in enumerate(vs):
            top_paths_for_i = recur(past_layer, i, memoizer)
            for (node_path, val) in top_paths_for_i:
                new_path = node_path + [node_layer_idx]
                new_val = val + v
                paths.append((new_path, new_val))

        seen = set()
        deduped_list = [x for x in paths if not (
            str(x) in seen or seen.add(str(x)))]
        deduped_list.sort(key=lambda x: x[1], reverse=True)
        cutoff = min(k, len(deduped_list))
        deduped_list = paths[:cutoff]
        # TODO: do we need to make a copy?
        # [d for d in deduped_list]
        memoizer[(layer, node_layer_idx)] = deduped_list
        return deduped_list

    top_k = recur(top_layer, 0, {})
    # seen = set()
    # deduped_list = [x for x in top_flat if not (str(x) in seen or seen.add(str(x)))]
    top_k.sort(key=lambda x: x[1], reverse=True)
    cutoff = min(k, len(top_k))
    sorted = top_k[:cutoff]
    return sorted
    # memoizer[(layer, node)] =
    # TODO: return

# We have to implement https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3009499/


def top_k_paths_to_end(G, start, end, k, weight='weight'):
    # Perform topological sort on the DAG
    topo_sort = list(nx.topological_sort(G))

    # Initialize dictionaries for storing the maximum path weights and paths
    max_weight = {node: float('-inf') for node in G}
    max_weight[start] = 0
    paths = {node: [] for node in G}
    paths[start] = [[start]]

    # Heap to maintain top k paths
    top_paths = []

    # Update weights and paths based on the topological order
    for node in topo_sort:
        for successor in G.successors(node):
            edge_weight = G[node][successor][weight]
            new_weight = max_weight[node] + edge_weight

            # Update only if the new weight is better
            if new_weight > max_weight[successor]:
                max_weight[successor] = new_weight
                paths[successor] = [path + [successor] for path in paths[node]]
            # Handle paths with equal weight
            elif new_weight == max_weight[successor]:
                paths[successor].extend([path + [successor]
                                        for path in paths[node]])

    # Only consider paths that end at the 'end' vertex
    for path in paths[end]:
        path_weight = sum(G[path[i]][path[i + 1]][weight]
                          for i in range(len(path) - 1))
        if len(top_paths) < k:
            heapq.heappush(top_paths, (path_weight, path))
        else:
            heapq.heappushpop(top_paths, (path_weight, path))

    # Sort the results based on weights in descending order before returning
    top_paths.sort(reverse=True, key=lambda x: x[0])
    return top_paths


def calculate_matmul_n_times(n_components, mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (1, k, d, d)
    """
    res = torch.zeros(mat_a.shape).to(mat_a.device)

    for i in range(n_components):
        mat_a_i = mat_a[:, i, :, :].squeeze(-2)
        mat_b_i = mat_b[0, i, :, :].squeeze()
        res[:, i, :, :] = mat_a_i.mm(mat_b_i).unsqueeze(1)

    return res


def calculate_matmul(mat_a, mat_b):
    """
    Calculate matrix product of two matrics with mat_a[0] >= mat_b[0].
    Bypasses torch.matmul to reduce memory footprint.
    args:
        mat_a:      torch.Tensor (n, k, 1, d)
        mat_b:      torch.Tensor (n, k, d, 1)
    """
    assert mat_a.shape[-2] == 1 and mat_b.shape[-1] == 1
    return torch.sum(mat_a.squeeze(-2) * mat_b.squeeze(-1), dim=2, keepdim=True)


def get_random_cutoff(t: str, size: int):
    if len(t) <= size:
        return t
    start_r = np.random.randint(0, len(t) - size)
    end = start_r + size
    return t[start_r:end]


def cosine_similarity_with_metric(a: npt.NDArray, b: npt.NDArray, metric: npt.NDArray):
    return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def pairwise_pearson_coefficient(A: npt.NDArray, B: npt.NDArray, eps=1e-8):
    """
    Compute the pairwise Pearson Correlation Coefficient between two matrices of features.

    Correlation is computed *row wise*

    From https://stackoverflow.com/questions/33650188/efficient-pairwise-correlation-for-two-matrices-of-features
    """
    assert A.shape[1] == B.shape[1], "Matrices must have the same number of features"
    assert len(A.shape) == 2 and len(B.shape) == 2, "Matrices must be 2D"
    A = A.T
    B = B.T
    mmap_A = np.memmap('/tmp/mmatA.dat', dtype='float32',
                       mode='w+', shape=A.shape)
    mmap_B = np.memmap('/tmp/mmatB.dat', dtype='float32',
                       mode='w+', shape=B.shape)
    mmap_A[:] = A[:]
    mmap_B = B[:]
    A = mmap_A
    B = mmap_B

    N = B.shape[0]
    p1 = N*np.dot(B.T, A)
    sA = A.sum(0)
    sB = B.sum(0)
    p2 = sA*sB[:, None]
    p3 = N*((B**2).sum(0)) - (sB**2)
    p4 = N*((A**2).sum(0)) - (sA**2)

    # Finally compute Pearson Correlation Coefficient as 2D array
    # print("DIVIDING BY", np.sqrt(p4*p3[:, None]))
    pcorr = ((p1 - p2) / (np.sqrt(p4*p3[:, None]) + eps))
    return pcorr


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

def restrict_to_related_vertex(lattice: List[npt.NDArray], layer: int, idx: int) -> List[npt.NDArray]:
    bellow = [] if layer < 2 else lattice[0:layer-1]
    prior = [] if layer == 0 else [lattice[layer-1][:, idx:idx+1]]
    above = [] if layer >= len(lattice) - 1 else lattice[layer+1:]
    curr = [] if layer == len(lattice) else [lattice[layer][idx:idx+1]]
    # print(bellow[0].shape, prior[0].shape, 
    # print(bellow[0].shape, prior[0].shape, curr[0].shape, above[0].shape, len(above))
    return bellow + prior + curr + above

def top_k_dag_paths(layers: List[np.ndarray], layer: int, neuron: int, k: int):
    r = restrict_to_related_vertex(layers, layer, neuron)
    # print(r)
    graph, source, sink, graph_layers_to_idx, node_layers_to_graph, most_pos = to_nx_graph(
        r)
    X = nx.shortest_simple_paths(graph, source, sink, weight='weight')
    # print(len(node_layers_to_graph))

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
        print(paths[-1])
        if counter == k-1:
            break
    return paths

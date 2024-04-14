from typing import List, Dict, Tuple
from scipy.stats import chi2_contingency
import torch
import numpy as np
import networkx as nx
import heapq
import numpy.typing as npt
from sklearn.metrics import mutual_info_score

# TODO: this is just a c and p rn

# Get the activations for the best dict features
# TODO: fix up


def get_weighting_for_layer(layer: int, n_layers: int, weight_decay=0.8, peak_at_layer_only=False):
    # TODO: SOMETHING CLEANER
    # if prior_layers_decay is None:
    #     prior_layers_decay = weight_decay
    r = np.ones(n_layers)
    r[layer] = 1
    G = weight_decay
    for i in range(layer):
        if peak_at_layer_only:
            r[i] = 1e-5
        else:
            r[i] = G ** (layer - i)
    for i in range(layer + 1, n_layers):
        if peak_at_layer_only:
            r[i] = 1e-5
        else:
            r[i] = G ** (i - layer)
        # r[i] = 0.0001
    # TODO: THE LAST LAYER SEEMS TO SELECTIVE... we need a smarter way of doing things than simply choosing highest
    # if layer != n_layers - 1:
        # r[-1] = 0.0
    r[layer] = 1
    return r


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


def cosine_similarity_with_metric(a: npt.NDArray, b: npt.NDArray, metric: npt.NDArray) -> npt.NDArray:
    return np.inner(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# def mutual_information(A: npt.NDArray, B: npt.NDArray):
#     c_xy = np.histogram2d(x, y, bins)[0]
#     g, p, dof, expected = chi2_contingency(c_xy, lambda_="log-likelihood")
#     mi = 0.5 * g / c_xy.sum()
#     return mi


# def get_mutual_info(A: npt.NDArray, B: npt.NDArray):
#     mutual_info_regression(


def pairwise_mutual_information(A: npt.NDArray, B: npt.NDArray):
    # Number of rows in the matrix
    n_rows = A.shape[0]
    
    # Initialize a 2D array to store mutual information values
    mi_matrix = np.zeros((n_rows, n_rows))
    
    # Compute pairwise mutual information
    for i in range(n_rows):
        for j in range(i + 1, n_rows):
            # Compute mutual information between row i and row j
            mi = mutual_info_score(A[i], B[j])
            
            # Store the value in the matrix
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    
    return mi_matrix

def pairwise_pearson_coefficient_abs(A: npt.NDArray, B: npt.NDArray, eps=1e-8):
    """
    Compute the pairwise Absolute Value of the Pearson Correlation Coefficient between two matrices of features.

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
    # return pcorr * (pcorr > 0)
    return np.abs(pcorr) # TODO: BETTER EXPLAIN FOR WHY ONLY POS

# TODO: SEP FILE
def pairwise_correlation_metric(A: npt.NDArray, B: npt.NDArray):
    return pairwise_mutual_information(A, B)

def restrict_to_related_vertex(lattice: List[npt.NDArray], layer: int, idx: int) -> List[npt.NDArray]:
    bellow = [] if layer < 2 else lattice[0:layer-1]
    prior = [] if layer == 0 else [lattice[layer-1][:, idx:idx+1]]
    above = [] if layer >= len(lattice) - 1 else lattice[layer+1:]
    curr = [] if layer == len(lattice) else [lattice[layer][idx:idx+1]]
    # print(bellow[0].shape, prior[0].shape,
    # print(bellow[0].shape, prior[0].shape, curr[0].shape, above[0].shape, len(above))
    return bellow + prior + curr + above

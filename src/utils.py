from typing import List, Dict, Tuple
from scipy.stats import chi2_contingency
import torch
import numpy as np
import networkx as nx
import heapq
import numpy.typing as npt
from sklearn.metrics import mutual_info_score

def separate_neg_pos(embeds: npt.NDArray):
    """
    Go from an embedding of [n_data, embed dim] --> [n_data, 2 * embed_dim]
    where negative and positive are separated out
    """
    result = np.repeat(embeds, 2, axis=-1)
    indices = np.arange(result.shape[-1])
    indices_pos = np.nonzero(indices % 2 == 0)[0]
    indices_neg = np.nonzero(indices % 2 == 1)[0]
    result[indices_pos] = (result[indices_pos] > 0) * result[indices_pos]
    result[indices_neg] = (result[indices_neg] < 0) * result[indices_pos]
    return result

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

def signaling_maximization(a: npt.NDArray[1], B: npt.NDArray[2],
                           max_bound=2.0, eps=1e-5, use_geometric=False) -> float:
    """
    Get the cutoff from a to B such that the entries in a which are above the cutoff maximize signaling in B
    """
    assert len(B.shape) == 2, "Expected a matrix for B"
    assert len(a.shape) == 1, "Expected a matrix for vector for a"
    assert a.shape[0] == B.shape[1], "Expected the same number of data points"
    B = B > eps

    n_features = B.shape[0]

    def geometric(s):
        if np.any(s == 0):
            return 0.0
        return np.exp(np.log(s).sum() / s.shape[-1])
    def arithmetic(s):
        return np.sum(s) / s.shape[-1]

    
    max_score = -1
    maximizing_cutoff = 0.0

    inner_int_steps = 5

    for v in range(0, int(max_bound) * inner_int_steps + 1):
        cutoff = v / inner_int_steps

        signaling = np.zeros((n_features))
        anz = a > cutoff
        a_nonzero = anz.nonzero()[0]
        a_cutoff = a[a_nonzero]
        for i in range(n_features):
            # We expect a_cutoff > 0 to be true for all a_cutoff
            p_i_and_j = np.logical_and(a_cutoff > 0, B[i, a_nonzero] > 0)

            # TODO: hrmm min gives commutativity but we may not want this
            p_j = a_cutoff.shape[-1]
            if p_j > 0:
                signaling[i] = p_i_and_j.sum() / p_j
        
        score = geometric(signaling) if use_geometric else arithmetic(signaling)
        if score > max_score:
            max_score = score
            maximizing_cutoff = cutoff

    return maximizing_cutoff


# TODO: smarter` cutoff per row. BUT QUANTIZATION!
# TODO: this can be made wayyyy faster...
def pairwise_signaling(A: npt.NDArray[2], B: npt.NDArray[2], cutoff_per_row=0.005):
    """
    Pairwise **directed** signaling from A to B
    """
    print(A.shape, B.shape)
    assert len(B.shape) == len(A.shape) == 2, "Expected a matrix"
    assert A.shape[1] == B.shape[1], "Expected the same number of samples"
    A = A > cutoff_per_row
    B = B > cutoff_per_row
    n_rows_A = A.shape[0]
    n_rows_B = B.shape[0]
    signaling = np.zeros((n_rows_A, n_rows_B))
    # n_items = A.shape[1]
    # prob_row = signaling.sum(axis=-1) / n_items

    for i in range(n_rows_A):
        for j in range(n_rows_B):
            # Look at
            p_i_and_j = np.logical_and(A[i], B[j])
            # TODO: hrmm min gives commutativity but we may not want this
            # TODO: THIS IS NOT COMMUTATIVE >:/
            p_i_j_min = A[i].sum()
            if p_i_j_min > 0:
                signaling[i, j] = p_i_and_j.sum() / p_i_j_min
    return signaling


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
    return np.abs(pcorr)  # TODO: BETTER EXPLAIN FOR WHY ONLY POS

# TODO: SEP FILE


def pairwise_correlation_metric(A: npt.NDArray, B: npt.NDArray):
    return pairwise_signaling(A, B)
    # return pairwise_mutual_information(A, B)


def restrict_to_related_vertex(lattice: List[npt.NDArray], layer: int, idx: int) -> List[npt.NDArray]:
    bellow = [] if layer < 2 else lattice[0:layer-1]
    prior = [] if layer == 0 else [lattice[layer-1][:, idx:idx+1]]
    above = [] if layer >= len(lattice) - 1 else lattice[layer+1:]
    curr = [] if layer == len(lattice) else [lattice[layer][idx:idx+1]]
    # print(bellow[0].shape, prior[0].shape,
    # print(bellow[0].shape, prior[0].shape, curr[0].shape, above[0].shape, len(above))
    return bellow + prior + curr + above

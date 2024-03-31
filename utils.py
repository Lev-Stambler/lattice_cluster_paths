from typing import List
import torch
import numpy as np
from einops import rearrange
import networkx as nx
import heapq
import transformer_lens
# TODO: this is just a c and p rn

# Get the activations for the best dict features
# TODO: fix up

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
        deduped_list = [x for x in paths if not (str(x) in seen or seen.add(str(x)))]
        deduped_list.sort(key=lambda x: x[1], reverse=True)
        cutoff = min(k, len(deduped_list))
        deduped_list = paths[:cutoff]
        # TODO: do we need to make a copy?
        memoizer[(layer, node_layer_idx)] = deduped_list# [d for d in deduped_list]
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
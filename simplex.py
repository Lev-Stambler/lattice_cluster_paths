from typing import List, Tuple
import graph
import networkx as nx
import numpy as np
import numpy.typing as npt
import utils

Face = Tuple[float, List[int]]


def _get_avg_clique_weight(weight_attrs, face: List[int]):
    total_weight = 0
    total_cons = 0
    # print(face)
    # TODO: IDEALLY WE NEVER FACE THIS!!
    # TODO: we do have this though... this is like a dead neuron
    if len(face) <= 1:
        return 0.0
    # if len(clique) > 15:
    #     return 0
    # print("GET AVG", len(clique))
    for i in range(len(face)):
        for j in range(i):
            total_cons += 1
            o = [face[i], face[j]]
            o.sort()
            o = tuple(o)
            total_weight += weight_attrs[o]
    return total_weight / (total_cons ** 2)


def _calc_across_layer_sim_score(clique_lower: Face,
                                 clique_upper: Face,
                                 inter_layer_corr: npt.NDArray[2]) -> float:
    """
    An **commutative** function (i.e. clique_lower and clique_upper are exchangeable)
    """
    total_corr = 0.0
    for cl in clique_lower[1]:
        for cu in clique_upper[1]:
            total_corr += inter_layer_corr[cl, cu]
    return total_corr


def find_high_weight_faces(correlations: npt.NDArray[2],
                           n_cliques_per_vertex=3, n_search_per_vertex=100,
                           correlation_cutoff=0.04) -> List[Face]:
    """
    Find the high weight cliques within a layer's correlation graph.
    The return is a list of tuples corresponding to clique weight and the indices
    """
    assert len(
        correlations.shape) == 2 and correlations.shape[0] == correlations.shape[1], "Must have a valid correlation matrix"

    corrs_cutoff = correlations * (correlations > correlation_cutoff)
    cliques_per_node = []
    n_nodes = correlations.shape[0]
    for i in range(len(corrs_cutoff)):
        corrs_cutoff[i, i] = 0
    G = nx.from_numpy_array(corrs_cutoff, edge_attr='weight')

    for N in range(n_nodes):
        # print("Getting clique centered at", N)
        cliques_per_node.append([])
        y = nx.find_cliques(G, nodes=[N])
        for i, conns in enumerate(y):
            # print(conns)
            cliques_per_node[-1].append(conns)
            if i >= n_search_per_vertex:
                break

    # Get top cliques per node
    top_cliques_per_node: List[Face] = []
    weight_attrs = nx.get_edge_attributes(G, 'weight')
    for node, cliques in enumerate(cliques_per_node):
        top_cliques_per_node.append([])
        # print("Looking at node", node)
        weights = np.array(
            [_get_avg_clique_weight(weight_attrs, c) for c in cliques])
        # cliques_per_node[node] = zip(weights, cliques)
        tops = np.argsort(weights)[::-1]
        n = min(len(tops), n_cliques_per_vertex)
        for i in range(n):
            # print(tops[i], cliques[tops[i]], weights[tops[i]])
            top_cliques_per_node[-1].append((weights[tops[i]],
                                            cliques[tops[i]]))

    flattened = [c for cs in top_cliques_per_node for c in cs]
    included_set = set()
    unique = [f for f in flattened
              if str(f[1]) not in included_set or included_set.add(str(f[1]))]

    return unique


def face_correlation_lattice(inter_layer_correlations: List[npt.NDArray[2]],
                               faces: List[List[Face]]) -> List[npt.NDArray[2]]:
    n_layers = len(faces)
    clique_lists = [
        np.memmap(f'/tmp/mmat_clique_corr_layer_{layer}.dat', dtype='float32',
                  mode='w+', shape=(len(faces[layer]), len(faces[layer + 1]))
                  )
        for layer in range(n_layers - 1)]
    # For every layer calculate the cliques
    for layer in range(n_layers - 1):
        for i, face_lower in enumerate(faces[layer]):
            for j, face_upper in enumerate(faces[layer + 1]):
                clique_lists[layer][i, j] = _calc_across_layer_sim_score(face_lower,
                                                                  face_upper,
                                                                  inter_layer_correlations[layer])
    return clique_lists

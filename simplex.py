from typing import List, Tuple
import graph
import networkx as nx
import numpy as np
import numpy.typing as npt
import utils
import kernel

Face = Tuple[float, List[int]]


# TODO: GEOMETRIC OR ARITHMETIC!!!???
def get_clique_score(weight_attrs, face: List[int]):
    total_weight = 1
    total_cons = 0
    # TODO: IDEALLY WE NEVER FACE THIS!!
    # TODO: we do have this though... this is like a dead neuron
    if len(face) <= 1:
        return 0.0
    # TODO: RM this when find smarter way...
    # if len(face) > 5:
    #     return 0
    for i in range(len(face)):
        for j in range(i):
            total_cons += 1
            o = [face[i], face[j]]
            o.sort()
            o = tuple(o)
            total_weight += weight_attrs[o]

    if total_weight == 0.0:
        return 0.0
    assert total_weight > 0.0
    # We want to bias towards smaller cliques
    return total_weight
    # return total_weight ** (1 / (total_cons))
    return total_weight / (total_cons)


def _calc_across_layer_sim_score(clique_lower: Face,
                                 clique_upper: Face,
                                 inter_layer_corr: npt.NDArray[2]) -> float:
    """
    An **commutative** function (i.e. clique_lower and clique_upper are exchangeable)

    We are looking for the **average** weight across the edges. This way, large cliques are not advantaged
    """
    total_corr = 1.0
    for cl in clique_lower[1]:
        for cu in clique_upper[1]:
            total_corr *= inter_layer_corr[cl, cu]
    if total_corr == 0.0:
        return 0.0
    assert total_corr > 0.0
    n_conns = len(clique_lower[1] * len(clique_upper[1]))
    # Get the *GEOMETRIC* average connection weight
    r = total_corr ** (1 / n_conns)
    return r
    # return total_corr / (len(clique_lower[1] * len(clique_upper[1])))


def sparsify_correlation_graph(G: nx.Graph, keep_neighbs_upper: int):
    weight_attrs = nx.get_edge_attributes(G, 'weight')
    for node in G.nodes:
        edges = [tuple(list(sorted(e))) for e in G.edges(node)]
        if len(edges) == 0:  # TODO: ahahaha 0
            continue
        es = [weight_attrs[edge] for edge in edges]
        es.sort(reverse=True)
        cutoff = es[min(keep_neighbs_upper, len(es) - 1)]
        to_remove = [edge for edge in edges if weight_attrs[edge] < cutoff]

        # G.remove_edge(
        for e in to_remove:
            G.remove_edge(*e)
        # (G.remove_edge(*edge) for edge in to_remove)
    d = [k[1] for k in G.degree()]
    return G


def find_high_weight_faces(correlations: npt.NDArray[2],
                           layer_activations: npt.NDArray[2],
                           n_cliques_per_vertex=5, n_search_per_vertex=200,
                           n_max_collect_per_vertex=10,
                           upper_neighbs=50,  # TODO: Find right param
                           correlation_cutoff=0.1) -> List[Face]:  # TODO: what is the right correlation cutoff? Somehow we want to **encourage** sparsity. Of course this is like layer dependent. Maybe we do something like only keep top K neighbors of every node
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
    G = nx.from_numpy_array(corrs_cutoff)  # , edge_attr='weight')
    G = sparsify_correlation_graph(G, upper_neighbs)
    weight_attrs = nx.get_edge_attributes(G, 'weight')

    for N in range(n_nodes):
        print("Getting clique centered at", N)
        cliques_per_node.append([])
        y = nx.find_cliques(G, nodes=[N])
        for i, conns in enumerate(y):

            # TODO: BATCH?
            # TODO: BATCH HELPER FUNC!!
            prods = np.ones(layer_activations.shape[0])
            # Check if the clique has support on layer_activations
            for n in conns:
                prods *= kernel.feature_prob(layer_activations, n)

            # TODO: only supp on 1??
            if np.any(prods > 0):
                cliques_per_node[-1].append(conns)

            # _get_avg_clique_weight(weight_attrs, cliques_per_node[-1][-1])
            if len(cliques_per_node[-1]) >= n_max_collect_per_vertex:
                break
            if i >= n_search_per_vertex:
                break
        print(
            f"Centered at {N} has {len(cliques_per_node[-1])} supported cliques")

    # Get top cliques per node
    top_cliques_per_node: List[Face] = []
    for node, cliques in enumerate(cliques_per_node):
        top_cliques_per_node.append([])
        # print("Looking at node", node)
        weights = np.array(
            [get_clique_score(weight_attrs, c) for c in cliques])
        # cliques_per_node[node] = zip(weights, cliques)
        tops = np.argsort(weights)[::-1]
        n = min(len(tops), n_cliques_per_vertex)
        for i in range(n):
            # print(tops[i], cliques[tops[i]], weights[tops[i]]) # TODO: ideally with the right metric its not even a per node thing
            # print(weights[tops[i]])
            top_cliques_per_node[-1].append((weights[tops[i]],
                                            cliques[tops[i]]))

    flattened = [c for cs in top_cliques_per_node for c in cs]
    included_set = set()
    unique = [f for f in flattened
              if str(f[1]) not in included_set or included_set.add(str(f[1]))]

    return unique

def face_correlation_for_layer(layer: int, faces_lower, faces_upper, layer_correlations):
    c = np.memmap(f'/tmp/mmat_clique_corr_layer_{layer}.dat', dtype='float32',
                  mode='w+', shape=(len(faces_lower), len(faces_upper))
                  )
    for i, fl in enumerate(faces_lower):
        for j, fu in enumerate(faces_upper):
            c[i, j] = _calc_across_layer_sim_score(fl,
                                                   fu,
                                                   layer_correlations)
    return c


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
        print("Correlating", layer, "to", layer + 1)
        for i, face_lower in enumerate(faces[layer]):
            for j, face_upper in enumerate(faces[layer + 1]):
                clique_lists[layer][i, j] = _calc_across_layer_sim_score(face_lower,
                                                                         face_upper,
                                                                         inter_layer_correlations[layer])
    return clique_lists

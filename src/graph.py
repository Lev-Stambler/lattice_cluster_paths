import networkx as nx
import copy
from typing import List
from numpy import np
from numpy.typing import npt

def sparsify_weighted_graph(G: nx.Graph, degree_upperbound: int):
    """ Sparsify a graph such that Deg(v) <= degree_upperbound for all v
    Importantly, we want to keep the highest weight edges
    """
    # Deep copy as we will be modifying G
    G = copy.deepcopy(G)
    weight_attrs = nx.get_edge_attributes(G, 'weight')
    for node in G.nodes:
        edges = [tuple(list(sorted(e))) for e in G.edges(node)]
        if len(edges) == 0:  # TODO: ahahaha 0
            continue
        es = [weight_attrs[edge] for edge in edges]
        es.sort(reverse=True)
        cutoff = es[min(degree_upperbound, len(es) - 1)]
        to_remove = [edge for edge in edges if weight_attrs[edge] < cutoff]

        # G.remove_edge(
        for e in to_remove:
            G.remove_edge(*e)
        # (G.remove_edge(*edge) for edge in to_remove)
    return G



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

def graph_from_correlations(corrs: npt.NDArray):
    assert len(corrs.shape) == 2 and corrs.shape[0] == corrs.shape[1]
    Is = np.arange(corrs.shape[0])
    corrs_mod = np.array(corrs)
    corrs_mod[Is, Is] = 0
    return nx.from_numpy_array(corrs_mod)

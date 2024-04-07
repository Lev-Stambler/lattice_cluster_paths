import numpy as np
import graph
import utils
import numpy.typing as npt
from typing import List, Callable
import networkx as nx


class Node:
    layer: int
    feature: int

    def __init__(self, layer, feature) -> None:
        self.layer = layer
        self.feature = feature


class SubLattice:
    layer_to_features: List[List[int]]


class CorrelationLattice:
    _correlations: List[npt.NDArray[2]]
    _corr_cutoffs: List[npt.NDArray[1]]
    lattice: List[npt.NDArray[2]]

    # TODO: variable degree!!!
    def __init__(self, correlations: List[npt.NDArray], target_degree=8) -> None:
        self._correlations = correlations
        self._target_degree = target_degree
        self._corr_cutoffs = self._get_corr_cutoff()
        self.lattice = [
            ((corr > np.expand_dims(
                self._corr_cutoffs[i], axis=-1)) * 1).astype(int)
            for i, corr in enumerate(correlations)]
        print(self.lattice[0].sum(axis=-1))

    def _get_corr_cutoff(self):
        corrs = self._correlations
        cutoffs = []
        for _, layer in enumerate(corrs):
            sorted = np.flip(np.sort(layer, axis=-1), axis=-1)
            cutoff = sorted[:, self._target_degree]
            assert cutoff.shape[0] == layer.shape[0]
            print(cutoff)
            cutoffs.append(cutoff)
        return cutoffs

    # def get_all_lattice_paths_pairwise(self, top_node: Node, bot_node: Node = None) -> None:
    #     to_top = utils.restrict_to_related_vertex(self.lattice, top_node.layer, top_node.feature)
    #     if bot_node is not None:
    #         bot_and_top = utils.restrict_to_related_vertex(to_top, bot_node.layer, bot_node.feature)
    #     else:
    #         bot_and_top = to_top

    #     G, source, sink, graph_idx_to_node_idx, node_idx_to_graph_idx, \
    #         most_pos_per_layer = graph._to_nx_graph(bot_and_top)

    #     print("G neighb test", list(G.neighbors(1)))
    #     top_idx = node_idx_to_graph_idx[top_node.layer][0] # We only have 1 idx due to the restriction
    #     bot_idx = source if bot_node is None else node_idx_to_graph_idx[bot_node.layer][0] # We only have 1 idx due to the restriction
    #     all_paths_G = list(nx.all_simple_paths(G, bot_idx, top_idx))
    #     all_paths_nodes =
    #     print(all_paths)

    def calc_whole_lattice(self):
        raise NotImplementedError
        pass


_Path = List[int]
_PathEquivalence = List[_Path]
_LayerToItems = List


class PathLattice():
    _corr_lattice: CorrelationLattice
    _equiv_thresh = 0.5
    _lattice: _LayerToItems[List[_PathEquivalence]]

    def __init__(self, corr_lattice: CorrelationLattice) -> None:
        self._corr_lattice = corr_lattice
        self._lattice = [None for i in range(len(self._corr_lattice.lattice))]

    def _paths_for_layer(self, layer: int):
        for i in range(layer, len(self._lattice) ):
            assert self._lattice[i] is not None, "We need the above lattice scores to be calculated first"
        corrs_in_layer = self._corr_lattice._correlations[layer]
        pass

    def _path_equivalence_rule(self) -> None:
        """
        The default rule is that there are >= `self._equiv_thresh * len(path)` equivalent features in a path to any other representative 
        in that equivalence path, then it is put into that equivalence class
        """
        pass

    def override_path_eq_rule(self, new_rule: Callable):
        self._path_equivalence_rule = new_rule

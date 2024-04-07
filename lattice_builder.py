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
    _equiv_classes: _LayerToItems[List[_PathEquivalence]]

    def __init__(self, corr_lattice: CorrelationLattice) -> None:
        self._corr_lattice = corr_lattice
        self._equiv_classes = [None for i in range(len(self._corr_lattice.lattice) + 1)]
        assert self._equiv_thresh >= 0.5, "The lattice is not well defined for equiv_thresh bellow 1/2"

    def _calc_paths_for_layer(self, layer: int):
        if layer == len(self._equiv_classes):
            n_last_items = self._corr_lattice._correlations[-1].shape[-1]
            # We have one representative for every equivalence class
            self._equiv_classes[-1] = np.expand_dims(np.arange(n_last_items, dtype=int), -1).tolist()
            return self._equiv_classes[-1]
        
        def update_eq_classes(new_path: List[int], eq_classes: List[_PathEquivalence]):
            in_existing = False
            for i, c in enumerate(eq_classes):
                for rep in c:
                    assert len(rep) == len(new_path) # TODO: del assert
                    n_same_path = sum([a == b
                        for (a, b) in zip(rep, new_path)])
                    # TODO THIS IS STRICT >=
                    if n_same_path > self._equiv_thresh * len(rep):
                        eq_classes[i].append(new_path)
                        in_existing = True
                        return eq_classes

            if not in_existing:
                eq_classes.append([new_path])
            return eq_classes

        
        for i in range(layer, len(self._equiv_classes)):
            assert self._equiv_classes[i] is not None, "We need the above lattice scores to be calculated first"
        corrs = self._corr_lattice._correlations
        corrs_in_layer = corrs[layer]
        upstream_eq_classes = self._equiv_classes[layer + 1]
        curr_eq_classes = []

        for i, ue in enumerate(upstream_eq_classes):
            for node, _ in enumerate(corrs_in_layer):
                for path in ue:
                    next_idx = path[0]
                    has_connection = self._corr_lattice.lattice[layer][node, next_idx] > 0
                    if has_connection:
                        curr_eq_classes = update_eq_classes([node] + path, curr_eq_classes)
        self._equiv_classes[layer] = curr_eq_classes
        return curr_eq_classes
    
    def load(self):
        n_layers = len(self._equiv_classes)
        for i in reversed(range(n_layers)):
            self._calc_paths_for_layer(i)

    def _path_equivalence_rule(self) -> None:
        """
        The default rule is that there are >= `self._equiv_thresh * len(path)` equivalent features in a path to any other representative 
        in that equivalence path, then it is put into that equivalence class
        """
        pass

    def override_path_eq_rule(self, new_rule: Callable):
        self._path_equivalence_rule = new_rule

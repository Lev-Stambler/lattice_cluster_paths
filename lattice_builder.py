import numpy as np
import numpy.typing as npt
from typing import List
    
class SubLattice:
    layer_to_features: List[List[int]]   

class Lattice:
    _correlations: List[npt.NDArray[2]]
    _corr_cutoffs: List[npt.NDArray[1]]
    lattice: List[npt.NDArray[2]]
    
    def __init__(self, correlations: List[npt.NDArray], target_degree=4) -> None:
        self._correlations = correlations
        self._target_degree = target_degree
        self._corr_cutoffs = self._get_corr_cutoff()
        self.lattice = correlations > np.expand_dims(self._corr_cutoffs, axis=-1)
        print(self.lattice)
    
    def _get_corr_cutoff(self):
        corrs = self._correlations
        target = self._target_degree
        cutoffs = []
        for _, layer in enumerate(corrs):
            sorted = np.flip(np.sort(layer, axis=-1), axis=-1)
            cutoff = sorted[:, self._target_degree]
            assert cutoff.shape[0] == layer.shape[0]
            print(cutoff)
            cutoffs.append(cutoff)
        return cutoffs
    
    def get_down_set(self, layer: int, feature: int) -> None:
        pass

    def calc_whole_lattice(self):
        raise NotImplementedError
        pass

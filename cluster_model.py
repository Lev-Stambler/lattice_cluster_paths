from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import numpy as np
import numpy.typings as npt
import torch
import transformer_lens

def get_optimal_layer_kmeans(model_lens: transformer_lens.HookedTransformer, tokenizer, dataset, layer: str, N_CLUTERS_UPPERBOUND=1_000) -> List[npt.NDArray[np.float64]]:
    """
    For a specific layer, find the optimal number of clusters: TODO document some references for this is actually done
    Then, return the found centroids for each cluster
    """
    raise NotImplementedError
    pass

def cluster_model_lattice(model_lens, layers_to_centroids: List[List[npt.NDArray[np.float64]]], distance_cutoff=float("inf")):
    """
    We will take a bit of a short cut here. Rather than passing *representatives* from each centroid to find the "strength" on the following centroids,
    we will pass the *center* of each centroid to the next layer. This is a simplification, but it should be a good starting point and quite a bit faster.

    distance_cutoff: If the distance between two centroids is greater than this, we will not consider them to be connected.
    """
    raise NotImplementedError
    pass

def find_max_flow():
    """
    Find the maximum flow through the lattice

    https://www.usenix.org/conference/atc18/presentation/gong We can find the top K max flows with the "Heavy Keeper" Algorithm
    """
    raise NotImplementedError
    pass

model_name = 'EleutherAI/pythia-70m'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")

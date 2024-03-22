from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
from typing import List
import numpy as np
import numpy.typing as npt
import torch
import transformer_lens

DEFAULT_DEVICE = 'cpu'


def forward_pass(model_lens: transformer_lens.HookedTransformer, t: str, layer: str) -> npt.NDArray[np.float64]:
    o = model_lens.run_with_cache(t)
    print(o)


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


def get_transformer(name: str, device=DEFAULT_DEVICE):
    """
    Get the transformer model from the name
    """
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = transformer_lens.HookedTransformer.from_pretrained(name).to(device)
    return model, tokenizer


def get_dataset(name: str):
    """
    Get the dataset from the name
    """
    return load_dataset(name)


def main():
    model_name = 'EleutherAI/pythia-160m'
    dataset_name = 'NeelNanda/pile-10k'

    model, tokenizer = get_transformer(model_name)
    ds = get_dataset(dataset_name)
    print(ds['train'][0])
    forward_pass(model, ds['train'][0]['text'], "")
    return

    layers_to_centroids = []
    for layer in model_lens.layers:
        centroids = get_optimal_layer_kmeans(model, tokenizer, dataset, layer)
        layers_to_centroids.append(centroids)
    lattice = cluster_model_lattice(model, layers_to_centroids)
    max_flow = find_max_flow(lattice)
    return max_flow


if __name__ == '__main__':
    main()

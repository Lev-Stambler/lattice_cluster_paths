import os
import pickle
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset, load_dataset
from typing import List
import numpy as np
import numpy.typing as npt
import torch
import transformer_lens
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score
import scipy


DEFAULT_DEVICE = 'cpu'
# TODO: not global var
N_DIMS = 512
SEED = 69_420
N_DATASIZE = 10_000
N_DATASIZE = 100

def kmeans_silhouette_method(dataset, n_clusters_min=2 * N_DIMS, n_clusters_max=10 * N_DIMS, skip=30):
    tests = range(n_clusters_min, n_clusters_max, skip)
    opt_sil = -1
    opt_clusters = None

    # TODO: more like bin search
    for n_clusters in tests:
        print(f"Trying {n_clusters} clusters")
        clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED)
        cluster_labels = clusterer.fit_predict(dataset)
        silhouette_avg = silhouette_score(dataset, cluster_labels)

        if silhouette_avg > opt_sil:
            print(
                f"Found better silhouette score: {silhouette_avg} with {n_clusters} clusters")
            opt_sil = silhouette_avg
            opt_clusters = n_clusters
    return opt_clusters, opt_sil


def forward_pass(model_lens: transformer_lens.HookedTransformer, t: str, layer: str) -> npt.NDArray[np.float64]:
    o = model_lens.run_with_cache(t)[1]
    return o[layer]

def get_optimal_layer_kmeans(model_lens: transformer_lens.HookedTransformer, tokenizer, dataset, layer: str) -> List[npt.NDArray[np.float64]]:
    """
    For a specific layer, find the optimal number of clusters: TODO document some references for this is actually done
    Then, return the found centroids for each cluster
    """
    print(
        f"Finding optimal number of clusters and such clusters for layer {layer}")
    ds_name = f'{layer}_dataset_embd__SEED_{SEED}__SIZE_{N_DATASIZE}.pkl'

    # TODO: DOES THIS WORK?
    if os.path.exists(ds_name):
        dataset_np = pickle.load(open(ds_name, 'rb'))
    else:
        dataset_np = [forward_pass(model_lens, t, layer) for t in dataset]
        pickle.dump(dataset_np, open(ds_name, 'wb'))
    clusters, _ = kmeans_silhouette_method(dataset_np)
    return clusters


def cluster_model_lattice(model_lens, layers_to_centroids: List[List[npt.NDArray[np.float64]]], distance_cutoff=float("inf")):
    """
    We will take a bit of a short cut here. Rather than passing *representatives* from each centroid to find the "strength" on the following centroids,
    we will pass the *center* of each centroid to the next layer. This is a simplification, but it should be a good starting point and quite a bit faster.

    distance_cutoff: If the distance between two centroids is greater than this, we will not consider them to be connected.
    """
    
    def score_cluster_to_next(cluster, next_clusters, distance_cutoff) -> List[float]:
        """
        Score the cluster to the next clusters.
        Set any score to 0 if the distance between the two centroids is greater than the distance_cutoff
        
        # TODO: DIFFERENT METRICS???
        """
        raise NotImplementedError
        pass

    cluster_scores = []
    for layer in range(len(layers_to_centroids) - 1):
        layer_cs = []
        for _, cluster in enumerate(layers_to_centroids[layer]):
            scores_to_next = score_cluster_to_next(cluster, layers_to_centroids[layer + 1], distance_cutoff)
            layer_cs.append(scores_to_next)
        cluster_scores.append(layer_cs)

    return cluster_scores

def find_max_flow(cluster_scores: List[List[List[float]]]):
    """
    Find the maximum flow through the lattice

    https://www.usenix.org/conference/atc18/presentation/gong We can find the top K max flows with the "Heavy Keeper" Algorithm
    """
    csgraph = np.array()
    node_idx = 0

    source = 0
    node_idx += 1
    n_clusters = sum([sum(len(cs) for cs in layer) for layer in cluster_scores])
    csr_matrix = np.zeros((n_clusters, n_clusters), dtype=int)

    

    for layer in range(len(cluster_scores), - 1):
        layer_start_idx = node_idx
        n_in_layer = len(cluster_scores[layer])
        for _, node_cs in enumerate(cluster_scores[layer]):
            for j, c in enumerate(node_cs):
                next_idx = layer_start_idx + n_in_layer + j
                csgraph[node_idx, next_idx] = c
            node_idx += 1

    sink_idx = node_idx
    # TODO: add source and sink connections
    # TODO: to CSR matrix

    scipy.sparse.csgraph.maximum_flow(csgraph, source, sink)
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
    model_name = 'EleutherAI/pythia-70m'
    dataset_name = 'NeelNanda/pile-10k'
    n_blocks = 6
    def get_block_label(i): return f'blocks.{i}.hook_resid_post'
    model, tokenizer = get_transformer(model_name)

    ds = get_dataset(dataset_name)
    # TODO: we will have to do more than th
    shuffled = ds.shuffle(seed=SEED)['train'][:N_DATASIZE]['text']
    ds = shuffled

    # forward_pass(model, ds['train'][0]['text'], "")

    clusters = []
    for i in range(n_blocks):
        c = get_optimal_layer_kmeans(model, tokenizer, ds, get_block_label(i))
        clusters.append(c)
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

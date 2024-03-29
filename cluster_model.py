import os
import pickle
from datasets import Dataset, load_dataset
from typing import List, Union
import numpy as np
import numpy.typing as npt
import torch
import transformer_lens
from sklearn.mixture import GaussianMixture
import hashlib
import networkx as nx
import utils


DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# TODO: not global var
N_DIMS = 512
SEED = 69_420
N_DATASIZE = 10_000
N_CLUSTERS_MIN = int(0.5 * N_DIMS)
N_CLUSTERS_MAX = 10 * N_DIMS
# TODO: CHANGE BACK TO 6/ MAKE THIS A PARAM
N_BLOCKS = 2
N_TOKENS_CUTOFF = 100

MODEL_NAME = 'EleutherAI/pythia-70m'
DATASET_NAME = 'NeelNanda/pile-10k'

DEBUG_N_DATASIZE = 70
DEBUG_N_CLUSTERS_MIN = 40
DEBUG_N_CLUSTERS_MAX = 41

# DEBUG_N_CLUSTERS_MIN = 10
# DEBUG_N_CLUSTERS_MAX = 20

DEBUG_N_BLOCKS = 3
DEBUG = True

if DEBUG:
    N_DATASIZE = DEBUG_N_DATASIZE
    N_CLUSTERS_MIN = DEBUG_N_CLUSTERS_MIN
    N_CLUSTERS_MAX = DEBUG_N_CLUSTERS_MAX
    N_BLOCKS = DEBUG_N_BLOCKS


def create_param_tag():
    m = hashlib.sha256()
    m.update(
        f'{MODEL_NAME}{DATASET_NAME}{SEED}{N_DATASIZE}{N_CLUSTERS_MIN}{N_CLUSTERS_MAX}{N_BLOCKS}'.encode())
    return m.hexdigest()[:40]


def similarity_metric(a: npt.NDArray, b: npt.NDArray):
    return np.sum(np.exp(np.multiply(a, b)))


def similarity_for_gmm(gmm: GaussianMixture, a: npt.NDArray, b: npt.NDArray):
    preds_a = gmm.predict_proba(a.expand_dims(0))
    preds_b = gmm.predict_proba(b.expand_dims(0))
    return similarity_metric(preds_a, preds_b)
    # return np.inner(a, b)


def get_save_tag(prepend: str):
    return f'metadata/{prepend}_{create_param_tag()}.pkl'


def get_block_base_label(i): return f'blocks.{i}'


def get_block_out_label(i): return f'{get_block_base_label(i)}.hook_resid_post'


def forward_on_block(model, block_idx: int, data: npt.NDArray):
    ret = model.blocks[block_idx](torch.tensor(list(data)).unsqueeze(
        0).unsqueeze(0).to(device=DEFAULT_DEVICE)).detach().cpu().numpy()
    return ret[0][0]


# TODO: optimize numb of clusters
def GMM_method(dataset: npt.NDArray, layer: int, n_clusters_min=N_CLUSTERS_MIN, n_clusters_max=N_CLUSTERS_MAX, skip=30) -> GaussianMixture:
    gm_name = get_save_tag(f'{layer}_clusters_GMM')
    if os.path.exists(gm_name):
        print("Loading GMM clusters from cache")
        gmm = pickle.load(open(gm_name, 'rb'))
        return gmm

    # TODO: bin search
    for n_clusters in range(n_clusters_min, n_clusters_max, skip):
        print(f"Trying {n_clusters} clusters")
        gm = GaussianMixture(n_components=n_clusters_min,
                             random_state=SEED).fit(dataset)
        pickle.dump(gm, open(gm_name, 'wb'))
        # silhouette_avg = silhouette_score(dataset, gm_name)
        # print(f"Silhouette score: {silhouette_avg}")
        return gm

# def kmeans_silhouette_method(dataset: npt.NDArray, layer: int, n_clusters_min=N_CLUSTERS_MIN, n_clusters_max=N_CLUSTERS_MAX, skip=30):
#     tests = range(n_clusters_min, n_clusters_max, skip)
#     opt_sil = -1
#     opt_clusters = None

#     cluster_name = get_save_tag(f'{layer}_clusters')
#     if os.path.exists(cluster_name):
#         print("Loading KMeans clusters from cache")
#         return pickle.load(open(cluster_name, 'rb'))

#     # TODO: more like bin search
#     for n_clusters in tests:
#         print(f"Trying {n_clusters} clusters")
#         # TODO: would be nice to be a non-mini-batch
#         clusterer = MiniBatchKMeans(n_clusters=n_clusters, random_state=SEED)
#         cluster_labels = clusterer.fit_predict(dataset)
#         silhouette_avg = silhouette_score(dataset, cluster_labels)

#         if silhouette_avg > opt_sil:
#             print(
#                 f"Found better silhouette score: {silhouette_avg} with {n_clusters} clusters")
#             opt_sil = silhouette_avg
#             opt_clusters = clusterer.cluster_centers_

#     pickle.dump(opt_clusters, open(cluster_name, 'wb'))
#     return opt_clusters


def forward_pass(model_lens: transformer_lens.HookedTransformer, t: str, layer: str) -> npt.NDArray[np.float64]:
    with torch.no_grad():
        return model_lens.run_with_cache(t)[1][layer]


# TODO: we should have better labeling!!
def get_per_layer_emb_dataset(model_lens: transformer_lens.HookedTransformer, dataset: Dataset, layers: List[str], use_save=True) -> List[npt.NDArray[np.float64]]:
    layers_out = []
    for layer in layers:
        ds_name = get_save_tag(f'{layer}_dataset_embd')

        # TODO: DOES THIS WORK?
        if os.path.exists(ds_name) and use_save:
            print("Loading dataset from cache")
            dataset_np = pickle.load(open(ds_name, 'rb'))
        else:
            # TODO: think about this in terms of flattening the dataset
            dataset_np_non_flat = [list(forward_pass(model_lens, t, layer).squeeze(
                0).detach().cpu().numpy()) for t in dataset]
            dataset_np = [d for ds in dataset_np_non_flat for d in ds]
            if use_save: pickle.dump(dataset_np, open(ds_name, 'wb'))
        layers_out.append(dataset_np)
    return np.stack(layers_out, axis=0)


def get_optimal_layer_gmm(dataset_np: npt.NDArray, layers: List[str], layer: str) -> List[npt.NDArray[np.float64]]:
    """
    For a specific layer, find the optimal number of clusters: TODO document some references for this is actually done
    Then, return the found centroids for each cluster
    """
    print(
        f"Finding optimal number of clusters and such clusters for layer {layer}")
    layer_idx = layers.index(layer)
    # ds_name = f'metadata/{layer}_dataset_embd__SEED_{SEED}__SIZE_{N_DATASIZE}.pkl'

    gm = GMM_method(dataset_np[layer_idx], layer)
    return gm


def cluster_model_lattice(model_lens, ds: npt.NDArray, gmms: List[GaussianMixture], similarity_cutoff=float("-inf")) -> List[List[List[float]]]:
    """
    We will take a bit of a short cut here. Rather than passing *representatives* from each centroid to find the "strength" on the following centroids,
    we will pass the *center* of each centroid to the next layer. This is a simplification, but it should be a good starting point and quite a bit faster.

    distance_cutoff: If the distance between two centroids is greater than this, we will not consider them to be connected.
    """
    save_name = get_save_tag('cluster_scores')
    if os.path.exists(save_name):
        print("Loading cluster scores from cache")
        return pickle.load(open(save_name, 'rb'))

    print("Getting cluster scores for lattice")
    # We want to have the outer index be the token, the inner index be the layer
    ds = ds.swapaxes(0, 1)

    def score_cluster_to_next(curr_layer_idx: int, next_layer_idx: int, metric_cutoff: float = None) -> List[float]:
        """
        Score the cluster to the next clusters.
        Set any score to 0 if the distance between the two centroids is greater than the distance_cutoff

        # TODO: DIFFERENT METRICS???
        # TODO: USING INNER PRODUCT RN
        """
        print("Getting scores for layers", curr_layer_idx, "to", next_layer_idx)
        HIGH_WEIGHT_PROB = 0.5

        to_next_layer_sim = np.zeros(
            (gmms[curr_layer_idx].n_components, gmms[next_layer_idx].n_components), dtype=int)

        # TODO: BATCHING is a good idea and not working right now
        # I think that I am missing a step here
        BS = 32
        for tok_idx in range(0, ds.shape[0], BS):
            tok = ds[tok_idx:tok_idx + BS]
            high_weight_currs = np.nonzero(gmms[curr_layer_idx].predict_proba(
                tok[:, curr_layer_idx]) > HIGH_WEIGHT_PROB)
            high_weight_nexts = np.nonzero(gmms[next_layer_idx].predict_proba(
                tok[:, next_layer_idx]) > HIGH_WEIGHT_PROB)
            # print("WITH BATCH", high_weight_currs, high_weight_nexts)
            for i in range(BS):
                col_idxs_curr = np.nonzero(high_weight_currs[0] == i)[0]
                col_idxs_next = np.nonzero(high_weight_nexts[0] == i)[0]

                high_weight_curr = high_weight_currs[1][col_idxs_curr]
                high_weight_next = high_weight_nexts[1][col_idxs_next]
                # print("INDEP", high_weight_curr, high_weight_next)
                for x in high_weight_curr:
                    for y in high_weight_next:
                        to_next_layer_sim[x, y] += 1
        return to_next_layer_sim

    cluster_scores = []
    for layer in range(len(gmms) - 1):
        scores_to_next = score_cluster_to_next(
            layer, layer + 1, similarity_cutoff)
        # TODO: into sparse matrix and then list??
        cluster_scores.append(scores_to_next)

    print("CLUSTER SCORES", cluster_scores)
    pickle.dump(cluster_scores, open(save_name, 'wb'))
    return cluster_scores


def to_nx_graph(cluster_scores: List[List[npt.NDArray]]) -> nx.DiGraph:
    SCALING_RESOLUTION = 1000
    node_idx = 0
    source = 0
    node_idx += 1
    # We need to account for all outgoing from the end
    n_clusters = sum([len(cs) for cs in cluster_scores]) + \
        len(cluster_scores[-1][0])
    eps = 1e-6
    most_neg = (min([min([min(c) for c in cs])
                for cs in cluster_scores])) + eps
    print(f"Most negative absolute value: {most_neg}")
    most_neg_abs = abs(most_neg)

    print(f"We have {n_clusters} clusters")

    G = nx.DiGraph()
    last_layer_start_idx = -1
    for layer in range(len(cluster_scores)):
        layer_start_idx = node_idx
        n_in_layer = len(cluster_scores[layer])
        if layer == len(cluster_scores) - 1:
            last_layer_start_idx = layer_start_idx + n_in_layer - 1
        for _, node_cs in enumerate(cluster_scores[layer]):
            for j, c in enumerate(node_cs):
                next_idx = layer_start_idx + n_in_layer + j
                # TODO: ROUNDING ISSUES?!!?
                # csgraph[node_idx, next_idx] = round(c)
                if c + most_neg_abs > 0:
                    w = round((c + most_neg_abs) * SCALING_RESOLUTION)
                # print("AAA", w, node_idx, next_idx)
                G.add_edge(node_idx, next_idx, weight=w)
            node_idx += 1

    sink = n_clusters

    for i, _ in enumerate(cluster_scores[0]):
        # Set source to first layer
        G.add_edge(source, i + 1, weight=1)
    for i in range(len(cluster_scores[-1])):
        G.add_edge(last_layer_start_idx + i, sink, weight=1)

    # nx.draw(G, with_labels=True, pos=nx.nx_pydot.graphviz_layout(G, prog='dot'))
    # plt.savefig("graph.png")
    return G, source, sink


# TODO: make this batched
def score_tokens_for_path(embd_dataset: npt.NDArray,
                                path: List[int], gmms: List[GaussianMixture],
                                score_weighting_per_layer: npt.NDArray, top_n=20):
    """
    embd_dataset: The outer index corresponds to the layer, the inner index corresponds to the token
    """
    # Set the outer index to the token
    token_per_layer = embd_dataset.swapaxes(0, 1)
    scores = np.zeros(len(token_per_layer))
    print(token_per_layer.shape, path)
    assert token_per_layer.shape[1] == len(path)

    for i, tok in enumerate(token_per_layer):
        score = 0
        for layer in range(len(path)):
            # TODO: change
            selector = np.zeros(gmms[layer].n_components)
            selector[path[layer]] = 1
            similarity_metric = np.inner(
                # TODO: we want to use a map from vertex to cluster
                selector, gmms[layer].predict_proba(
                    tok[layer].reshape(1, -1)
                )
            )  # TODO: SUPER SUPER GETHOT
            score += similarity_metric * score_weighting_per_layer[layer]
        scores[i] = score

    return scores
    # sorted_scores = np.argsort(scores)[::-1]
    # top = []
    # for i in range(top_n):
        # idx = sorted_scores[i]
        # top.append(
            # (token_to_original_ds[idx], token_to_pos_original_ds[idx], scores[idx]))
    # return top


def find_max_weight(cluster_scores: List[List[npt.NDArray]], K=100):
    """
    Find the maximum flow through the lattice

    https://www.usenix.org/conference/atc18/presentation/gong We can find the top K max flows with the "Heavy Keeper" Algorithm
    """
    G, source, sink = to_nx_graph(cluster_scores)
    paths = utils.top_k_paths_to_end(G, source, sink, K)
    # paths = utils.find_top_k_paths(G, source, sink, K)
    print("GOT PATHS", paths)
    return None, G


def get_transformer(name: str, device=DEFAULT_DEVICE):
    """
    Get the transformer model from the name
    """
    # tokenizer = AutoTokenizer.from_pretrained(name)
    model = transformer_lens.HookedTransformer.from_pretrained(name).to(device)
    tokenizer = model.tokenizer
    return model, tokenizer


def get_dataset(name: str):
    """
    Get the dataset from the name
    """
    return load_dataset(name)


class Decomposer:
    gmms: List[GaussianMixture]
    lattice_scores: List[List[List[float]]]

    def __init__(self, model_lens, dataset: Dataset, layers: List[str], similarity_cutoff=19):
        self.model_lens = model_lens
        self.dataset = dataset
        self.layers = layers
        self.similarity_cutoff = similarity_cutoff
        self.gmms = []
        self.lattice_scores = None

    def load(self):
        for i in range(len(self.layers)):
            self.gmms.append(get_optimal_layer_gmm(
                self.dataset, self.layers, self.layers[i]))
        self.lattice_scores = cluster_model_lattice(
            self.model_lens, self.dataset, self.gmms, self.similarity_cutoff)

    def score(self, to_score: List[str], score_path = [8, 57, 89], embeds: Union[npt.NDArray, None] = None) -> List[List[float]]:
        if embeds is None:
            embeds = get_per_layer_emb_dataset(
                self.model_lens, to_score, self.layers, use_save=False)
        ds = to_score
        token_to_original_ds = []
        token_to_pos_original_ds = []

        for i, d in enumerate(ds):
            tokenized = self.model_lens.to_tokens(d)[0]
            # print(tokenized)
            # return
            for j in range(len(tokenized)):
                token_to_original_ds.append(i)
                token_to_pos_original_ds.append(j)

        print("DS", embeds.shape, len(token_to_original_ds))
        assert len(token_to_original_ds) == len(
            token_to_pos_original_ds) == embeds.shape[1]
        scores = score_tokens_for_path(
            embd_dataset=embeds, path=score_path,
            gmms=self.gmms, score_weighting_per_layer=np.array([1, 1, 1]), top_n=20)
        item_to_scores = {}
        for i in range(len(scores)):
            item = token_to_original_ds[i]
            if item not in item_to_scores:
                item_to_scores[item] = []
            item_to_scores[item].append(scores[i])
        # Assuming that the scores are "dense" in how they were added, we have a list
        ks = sorted(list(item_to_scores.keys()))
        final_scores = []
        for k in ks:
            s = item_to_scores[k]
            final_scores.append(s)
        
        return final_scores


# def get_par

def main():
    model, tokenizer = get_transformer(MODEL_NAME)

    ds = get_dataset(DATASET_NAME)
    # TODO: we will have to do more than th
    shuffled = ds.shuffle(seed=SEED)['train'][:N_DATASIZE]['text']
    ds = shuffled
    # forward_pass(model, ds['train'][0]['text'], "")

    gmms = []
    labs = [get_block_out_label(i) for i in range(N_BLOCKS)]
    ds_emb = get_per_layer_emb_dataset(model, ds, labs)
    for i in range(N_BLOCKS):
        gmm = get_optimal_layer_gmm(
            ds_emb, labs, get_block_out_label(i))
        gmms.append(gmm)

    lattice = cluster_model_lattice(model, ds_emb, gmms, similarity_cutoff=19)
    max_flow = find_max_weight(lattice)

    token_to_original_ds = []
    token_to_pos_original_ds = []

    for i, d in enumerate(ds):
        tokenized = model.to_tokens(d)[0]
        # print(tokenized)
        # return
        for j in range(len(tokenized)):
            token_to_original_ds.append(i)
            token_to_pos_original_ds.append(j)

    print("DS", ds_emb.shape, len(token_to_original_ds))
    assert len(token_to_original_ds) == len(
        token_to_pos_original_ds) == ds_emb.shape[1]

    highest = score_tokens_for_path(
        token_to_pos_original_ds=token_to_pos_original_ds,
        token_to_original_ds=token_to_original_ds,
        embd_dataset=ds_emb, path=[8, 57, 89],
        gmms=gmms, score_weighting_per_layer=np.array([1, 1, 1]), top_n=20)
    print(highest)
    return max_flow, highest


if __name__ == '__main__':
    main()

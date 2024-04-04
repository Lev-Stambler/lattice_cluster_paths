import os
from scipy.stats import pearsonr
import pickle
from datasets import Dataset, load_dataset
from typing import List, Union, Dict
import numpy as np
import numpy.typing as npt
import torch
import transformer_lens
import metric
import hashlib
import json
import utils

# MixtureModel = KMeansMixture
# Use RBF Kernel https://en.wikipedia.org/wiki/Radial_basis_function_kernel


DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEFAULT_DEVICE = 'cpu'

# TODO: separate PARAMS file?
MODEL_NAME = 'EleutherAI/pythia-70m'
DATASET_NAME = 'NeelNanda/pile-10k'
MIXTURE_MODEL_TYPE = "KMenas"

N_DIMS = 512
SEED = 69_420

DEBUG = False

if DEBUG:
    N_DATASIZE = 200
    N_CLUSTERS_MIN = 20
    N_CLUSTERS_MAX = 21
    N_BLOCKS = 6
    STRING_SIZE_CUTOFF = 200
else:
    N_DATASIZE = 800
    # N_CLUSTERS_MIN = int(0.5 * N_DIMS)
    # N_CLUSTERS_MAX = 10 * N_DIMS
    N_CLUSTERS_MIN = N_DIMS
    N_CLUSTERS_MAX = N_DIMS + 1
    N_BLOCKS = 6
    STRING_SIZE_CUTOFF = 700


def metadata_json():
    return {
        'SEED': SEED,
        'MIXTURE_MODEL_TYPE': MIXTURE_MODEL_TYPE,
        'N_DATASIZE': N_DATASIZE,
        'N_CLUSTERS_MIN': N_CLUSTERS_MIN,
        'N_CLUSTERS_MAX': N_CLUSTERS_MAX,
        'N_BLOCKS': N_BLOCKS,
        'N_STRING_SIZE_CUTOFF': STRING_SIZE_CUTOFF
    }


def create_param_tag():
    m = hashlib.sha256()
    m.update(json.dumps(metadata_json()).encode('utf-8'))
    return m.hexdigest()[:40]


def similarity_metric(a: torch.Tensor, b: torch.Tensor):
    return torch.sum(torch.exp(torch.multiply(a, b)))


def score_for_neuron(self, to_score: List[str], primary_layer: int,
                     score_path: List[int],
                     top_n=100, primary_cutoff_mult_factor=3,
                     BS=128,
                     embeds: Union[npt.NDArray, None] = None,
                     weighting_per_layer=None) -> List[int]:
    raise NotImplementedError("This is not implemented yet for non mixture based")
    top_for_primary = top_n * primary_cutoff_mult_factor
    embeds, tokens_to_ds, _ = self._get_ds_metadata(
        to_score, embeds)
    max_n_tokens = embeds.shape[1]

    scores: List[(int, float)] = []

    for i in range(0, max_n_tokens, BS):
        top_idx = min(i + BS, max_n_tokens)
        e = embeds[:, i:top_idx]
        gmm_scores = score_for_gmm(
            self.gmms[primary_layer], e[primary_layer])
        to_add = zip(
            range(i, top_idx),
            list(gmm_scores[:, score_path[primary_layer]])
        )
        to_add = list(to_add)
        scores = scores + to_add

    top_by_score = sorted(scores, key=lambda x: x[1], reverse=True)[
        :top_for_primary]

    # Now, we want to get sort the top N by the path
    # TODO: weighting?

    top_score_idx = list([tokens_to_ds[x[0]] for x in top_by_score])
    included = set()
    # Dedup
    top_score_idx = [x for x in top_score_idx if not (
        x in included or included.add(x))]

    top_tokens = [x[0] for x in top_by_score]
    print("TOP TOKENS", len(top_tokens), top_tokens)
    embeds_for_top = embeds[:, top_tokens]
    ds_scored = [to_score[i] for i in top_score_idx]
    print("TOP SCORE IDX", top_score_idx)
    print("DS SCORED", len(ds_scored))
    print("EMB FOR TOP", embeds_for_top.shape)
    return top_tokens, top_score_idx, self.score(ds_scored, score_path, weighting_per_layer=weighting_per_layer)


def get_and_prepare_save_tag(prepend: str):
    if not os.path.exists('metadata'):
        os.mkdir('metadata')
    if not os.path.exists(f'metadata/{create_param_tag()}'):
        os.mkdir(f'metadata/{create_param_tag()}')
        json.dump(metadata_json(), open(
            f'metadata/{create_param_tag()}/metadata.json', 'w'))
    return f'metadata/{create_param_tag()}/{prepend}.pkl'


def get_block_base_label(i): return f'blocks.{i}'


def get_block_out_label(i): return f'{get_block_base_label(i)}.hook_resid_post'


def forward_on_block(model, block_idx: int, data: npt.NDArray):
    ret = model.blocks[block_idx](torch.tensor(list(data)).unsqueeze(
        0).unsqueeze(0).to(device=DEFAULT_DEVICE)).detach().cpu().numpy()
    return ret[0][0]


def forward_pass(model_lens: transformer_lens.HookedTransformer, t: str, layer: str) -> torch.Tensor:
    with torch.no_grad():
        return model_lens.run_with_cache(t)[1][layer]

# TODO: we should have better labeling!!

def get_per_layer_emb_dataset(model_lens: transformer_lens.HookedTransformer, dataset: Dataset, layers: List[str], use_save=True) -> List[npt.NDArray]:
    layers_out = []
    for layer in layers:
        ds_name = get_and_prepare_save_tag(f'{layer}_dataset_embd')

        # TODO: DOES THIS WORK?
        if os.path.exists(ds_name) and use_save:
            print("Loading dataset from cache")
            dataset_np = pickle.load(open(ds_name, 'rb'))
        else:
            # TODO: think about this in terms of flattening the dataset
            dataset_np_non_flat = [list(forward_pass(model_lens, t, layer).squeeze(
                0).detach().cpu().numpy()) for t in dataset]
            dataset_np = [d for ds in dataset_np_non_flat for d in ds]
            pickle.dump(dataset_np, open(ds_name, 'wb'))
        layers_out.append(dataset_np)
    return np.stack(layers_out, axis=0)


def cluster_model_lattice(ds: List[npt.NDArray]) -> List[List[List[float]]]:
    """
    We will take a bit of a short cut here. Rather than passing *representatives* from each centroid to find the "strength" on the following centroids,
    we will pass the *center* of each centroid to the next layer. This is a simplification, but it should be a good starting point and quite a bit faster.

    distance_cutoff: If the distance between two centroids is greater than this, we will not consider them to be connected.
    """
    save_name = get_and_prepare_save_tag('cluster_scores')
    if os.path.exists(save_name):
        print("Loading cluster scores from cache")
        return pickle.load(open(save_name, 'rb'))

    print("Getting cluster scores for lattice")

    # Use mm
    # TODO: parameterize by N_DIMS
    probs_for_all_layers = [
        np.memmap(f'/tmp/mmat_prob_layer_{layer}.dat', dtype='float32',
                  mode='w+', shape=(N_DIMS, ds.shape[1]))
        for layer in range(ds.shape[0])]
    for l in probs_for_all_layers:
        l[:] = 0.0
    print("Set all initial to 0")

    # TODO: we might want to batch this...
    for i in range(N_BLOCKS):
        ds_mmep = np.memmap(
            f'/tmp/mmat_ds_{i}.dat', dtype='float32', mode='w+', shape=ds[i].shape)
        ds_mmep[:] = ds[i]
        probs_for_all_layers[i][:] = metric.predict_proba(
            ds_mmep, batch_size=8_092).T
        print("Set predictions for layer", i)
    print("Set all probs with predictions")

    def score_cluster_to_next(curr_layer_idx: int, next_layer_idx: int) -> List[float]:
        coeffs = utils.pairwise_pearson_coefficient(
            probs_for_all_layers[curr_layer_idx], probs_for_all_layers[next_layer_idx])
        print("COEFF STUFF", probs_for_all_layers[curr_layer_idx].max(), probs_for_all_layers[curr_layer_idx].min(),
              coeffs.shape, coeffs.min(), coeffs.max())
        return coeffs

    cluster_scores = []
    for layer in range(len(ds) - 1):
        print(f"Scoring layer {layer}")
        scores_to_next = score_cluster_to_next(
            layer, layer + 1)
        # TODO: into sparse matrix and then list??
        cluster_scores.append(scores_to_next)

    print("CLUSTER SCORES", cluster_scores)
    pickle.dump(cluster_scores, open(save_name, 'wb'))
    return cluster_scores


def cutoff_lattice(lattice: List[List[List[float]]], related_cutoff=1):
    print(lattice[0].sum())
    r = [(layer > related_cutoff) * layer for layer in lattice]
    print(r[0].sum())
    return r

def score_tokens_for_path(embd_dataset: npt.NDArray,
                          path: List[int],
                          score_weighting_per_layer: npt.NDArray, BS=128):
    """
    embd_dataset: The outer index corresponds to the layer, the inner index corresponds to the token
    """
    # Set the outer index to the token
    token_per_layer = embd_dataset.swapaxes(0, 1)
    scores = np.zeros(len(token_per_layer))
    print(token_per_layer.shape, path)
    assert token_per_layer.shape[1] == len(path)

    for i in range(0, token_per_layer.shape[0], BS):
        top_idx = min(i + BS, token_per_layer.shape[0])
        for layer in range(len(path)):
            local_scores = metric.predict_proba(
                token_per_layer[i:top_idx, layer])
            local_scores = local_scores[:, path[layer]]
            # TODO: *integrate ideas of just getting overall log score*
            scores[i:top_idx] += score_weighting_per_layer[layer] * np.log(local_scores)
    return scores


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
    lattice_scores: List[List[List[float]]]

    def __init__(self, model_lens, dataset: Dataset, layers: List[str]):
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        print(f"Creating decomposer with parameter hash {create_param_tag()}")
        self.model_lens = model_lens
        # TODO: cutoff in random position

        self.dataset = [utils.get_random_cutoff(
            d, STRING_SIZE_CUTOFF) for d in dataset]
        self.layers = layers
        self.lattice_scores = None
        self.labs = [get_block_out_label(i) for i in range(N_BLOCKS)]
        self.ds_emb = get_per_layer_emb_dataset(
            self.model_lens, self.dataset, self.labs)

    def load(self):
        self.lattice_scores = cluster_model_lattice(
            self.ds_emb)

    def _get_ds_metadata(self, ds: List[str], embeds: npt.NDArray = None):
        if embeds is None:
            embeds = get_per_layer_emb_dataset(
                self.model_lens, ds, self.layers, use_save=False)
        token_to_original_ds = []
        token_to_pos_original_ds = []

        for i, d in enumerate(ds):
            tokenized = self.model_lens.to_tokens(d)[0]
            # print(tokenized)
            # return
            for j in range(len(tokenized)):
                token_to_original_ds.append(i)
                token_to_pos_original_ds.append(j)

        assert len(token_to_original_ds) == len(
            token_to_pos_original_ds) == embeds.shape[1]
        return embeds, token_to_original_ds, token_to_pos_original_ds

    def score_for_neuron(self, to_score: List[str], primary_layer: int,
                         score_path: List[int],
                         top_n=100, primary_cutoff_factor=3,
                         BS=1,
                         embeds: Union[torch.Tensor, None] = None,
                         weighting_per_layer=None) -> List[int]:
        """
        """
        return score_for_neuron(self, to_score, primary_layer, score_path,
                                top_n=top_n, primary_cutoff_mult_factor=primary_cutoff_factor,
                                BS=BS,
                                embeds=embeds, weighting_per_layer=weighting_per_layer)

    def score(self, to_score: List[str], score_path: List[int], embeds: Union[npt.NDArray, None] = None, weighting_per_layer=None, BS=128) -> List[List[float]]:
        """
        Get the scores for the tokens in the dataset
        """
        if weighting_per_layer is None:
            weighting_per_layer = np.ones(N_BLOCKS)

        embeds, token_to_original_ds, _ = self._get_ds_metadata(
            to_score, embeds)
        print("GOT EMBEDS", embeds.shape)
        scores = score_tokens_for_path(
            embd_dataset=embeds, path=score_path,
            score_weighting_per_layer=weighting_per_layer, BS=BS)
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

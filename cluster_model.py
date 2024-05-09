import os
import pickle
from datasets import Dataset, load_dataset
from typing import List, Union, Dict
import numpy as np
import numpy.typing as npt
import torch
import kernel
import params as paramslib
from model import TransformerModel, forward_hooked_model, load_model
import utils
import visualization
import graph
import json

from params import InterpParams, LatticeParams

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# DEFAULT_DEVICE = 'cpu' if torch.cuda.is_available() else 'cpu'


def get_top_scores(self, dataset: List[str], path: List[int],
                   layer: int, weighting_per_layer, embds=None, top_n=100):
    model = self.model_lens
    BOS_TOKEN = '||BOS||'
    score_path = path
    scores = self.score(
        dataset,
        layer=layer,
        score_path=score_path,
        weighting_per_layer=weighting_per_layer,
        use_log_scores=True,
        embeds=embds
    )

    # overall_scores = [[tok_score[0] for tok_score in s] for s in scores]
    scores_per_token_set = np.array([max(s) for s in scores])
    top_args = np.argsort(scores_per_token_set)[::-1][:top_n]
    # TODO: BOS?
    tokens = [[BOS_TOKEN] + [model.tokenizer.decode(t) for t in model.tokenizer(d)[
        'input_ids']] for d in dataset]
    tokens_reord = [tokens[i] for i in top_args]

    scores_reord = [scores[i] for i in top_args]
    return tokens_reord[:top_n], scores_reord[:top_n]

# TODO: by different model parameterize?


def get_layers_emb_dataset(model: TransformerModel, dataset: Dataset, layers: List[int], params: paramslib.InterpParams, use_save=True) -> List[npt.NDArray]:

    all_finished = params.get_and_prepare_data_save_tag('all_finished_embd')

    def mmat_file(layer: int):
        return params.get_and_prepare_data_save_tag(f'mmat_t_layer_total_{layer}.dat') \
            if use_save else f'/tmp/mmat_t_layer_total_{layer}.dat'
    if use_save and os.path.exists(all_finished):
        print("Loading dataset from cache")
        all_layers = []
        for i, _ in enumerate(layers):
            all_layers.append(
                np.memmap(mmat_file(i), dtype='float32', mode='r'))
        # We don't store the shape, so we need to reshape to the original shape
        # TODO: maybe store the shape instead?
        for i in range(len(all_layers)):
            all_layers[i] = all_layers[i].reshape(-1, params.model_n_dims)
        return all_layers

    all_outs = [[] for _ in layers]
    with torch.no_grad():
        BS = 1
        for t in range(0, len(dataset), BS):
            # if t % 200 == 0:
            top_idx = min(t + BS, len(dataset))
            d = dataset[t:top_idx]
            # torch.cuda.empty_cache()
            outs = forward_hooked_model(model, d)[1]

            for i, l in enumerate(layers):
                tens = outs[l]
                all_outs[i] += list(tens.cpu().numpy().reshape(-1,
                                    params.model_n_dims))
                del tens
    outs_np = [
    ]
    for l, _ in enumerate(layers):
        total_n_toks = len(all_outs[l])
        mmemap_name = mmat_file(l)
        out_np = np.memmap(mmemap_name, dtype='float32',
                           mode='w+', shape=(total_n_toks, params.model_n_dims))
        BS = 8
        for i in range(0, total_n_toks, BS):
            top_idx = min(i + BS, total_n_toks)
            out_np[i:top_idx, :] = np.array(all_outs[l][i:top_idx])
        outs_np.append(out_np)
    print("Finished and saving to file\n")
    if use_save:
        f = open(all_finished, 'w')
        f.write('done')
        f.close()
    return outs_np


def cluster_model_lattice(ds: List[npt.NDArray], params: paramslib.InterpParams) -> List[List[List[float]]]:
    """
    We will take a bit of a short cut here. Rather than passing *representatives* from each centroid to find the "strength" on the following centroids,
    we will pass the *center* of each centroid to the next layer. This is a simplification, but it should be a good starting point and quite a bit faster.

    distance_cutoff: If the distance between two centroids is greater than this, we will not consider them to be connected.
    """
    save_name = params.get_and_prepare_correlation_save_tag('cluster_scores')
    if os.path.exists(save_name):
        print("Loading cluster scores from cache")
        r = pickle.load(open(save_name, 'rb'))
        return r

    print("Getting cluster scores for lattice")

    # Use mm
    # TODO: parameterize by N_DIMS
    probs_for_all_layers = [
        np.memmap(f'/tmp/mmat_prob_layer_{layer}.dat', dtype='float32',
                  mode='w+', shape=(params.model_n_dims * 2, ds[layer].shape[0]))  # * 2 as each dimension has a +1 and -1
        for layer in range(len(ds))]
    for l in probs_for_all_layers:
        l[:] = 0.0
    print("Set all initial to 0")

    # TODO: we might want to batch this...
    for i in range(params.n_blocks):
        ds_mmep = np.memmap(
            f'/tmp/mmat_ds_{i}.dat', dtype='float32', mode='w+', shape=ds[i].shape)
        print("Trying to set dataset for layer", i)
        ds_mmep[:] = ds[i]
        print("Set dataset for layer", i, "Getting proba")
        probs_for_all_layers[i][:] = kernel.predict_proba(
            ds_mmep, batch_size=2_048 * 8).T
        print("Set predictions for layer", i)
    print("Set all probs with predictions")

    def score_cluster_to_next(curr_layer_idx: int, next_layer_idx: int) -> List[float]:
        coeffs = utils.pairwise_correlation_metric(
            probs_for_all_layers[curr_layer_idx], probs_for_all_layers[next_layer_idx])
        return coeffs

    cluster_scores = []
    for layer in range(len(ds) - 1):
        print(f"Scoring layer {layer}")
        scores_to_next = score_cluster_to_next(
            layer, layer + 1)
        # TODO: into sparse matrix and then list??
        cluster_scores.append(scores_to_next)

    pickle.dump(cluster_scores, open(save_name, 'wb'))
    return cluster_scores


def cutoff_lattice(lattice: List[List[List[float]]], related_cutoff=1):
    r = [(layer > related_cutoff) * layer for layer in lattice]
    return r


# TODO: this is no longer log?
def log_score_tokens_for_path(embd_dataset: List[npt.NDArray],
                              path: List[int],
                              score_weighting_per_layer: npt.NDArray, layer: int, BS=1_024*128, ignore_weights=False):
    """
    embd_dataset: The outer index corresponds to the layer, the inner index corresponds to the token
    """
    # Set the outer index to the token
    # token_per_layer = embd_dataset.swapaxes(0, 1)
    n_tokens = embd_dataset[0].shape[0]
    # scores = np.ones(n_tokens)
    n_layers = len(path)
    ret_scores = np.ones(n_tokens)
    assert len(embd_dataset) == len(path)

    # TODO: lazy load this or save it for use for later
    def get_cutoff_for_layer(neuron: int, layer_to_cutoff: int):
        fs = kernel.feature_prob(embd_dataset[layer_to_cutoff], neuron)
        nonzeros = fs[np.where(fs > 0)]
        if len(nonzeros) == 0:
            return 0.0
        return sum(nonzeros) / len(nonzeros)

    cutoffs = [
        get_cutoff_for_layer(n, i) for i, n in enumerate(path)
    ]

    for i in range(0, n_tokens, BS):
        # if i % (BS * 10) == 0:
        #     print("Scoring on", i, "of", n_tokens)
        top_idx = min(i + BS, n_tokens)
        for curr_layer in range(len(path)):
            local_scores = kernel.feature_prob(
                embd_dataset[curr_layer][i:top_idx], path[curr_layer])

            # Make sure that we never multiply by a negative number
            # Negative just means that we are anti-correlated
            # TODO: JUST 0 / 1?
            # local_scores = local_scores * (local_scores > 0.0)
            if not ignore_weights:
                ret_scores[i:top_idx] *= local_scores ** score_weighting_per_layer[curr_layer]
            else:
                ret_scores[i:top_idx] *= local_scores if layer == curr_layer else (
                    # ARGHHGHGHG would be so usefule to have a quantized model here...
                    # TODO: lets make sure to look into that...
                    local_scores > cutoffs[curr_layer])  # TODO: should we think about **segmented regions**?
            # local_scores
    return ret_scores


def get_dataset(name: str):
    """
    Get the dataset from the name
    """
    return load_dataset(name)


def correlate_internal_to_layer(embeds: List[npt.NDArray], params: InterpParams, use_saved=True):
    all_corrs = []
    for layer, embed in enumerate(embeds):
        f = params.get_and_prepare_correlation_save_tag(
            f'layer_{layer}_internal_corr')
        if use_saved and os.path.exists(f):
            print("Using saved correlation for layer", layer)
            fo = open(f, 'rb')
            r = pickle.load(fo)
            fo.close()
            all_corrs.append(r)
            continue

        prob_for_layer = np.memmap(f'/tmp/mmat_prob_layer_{layer}.dat', dtype='float32',
                                   mode='w+', shape=(params.model_n_dims * 2, embed.shape[0]))  # * 2 as each dimension has a +1 and -1
        prob_for_layer[:] = 0.0
        # dataset_mmep = np.memmap(
        #     f'/tmp/mmat_ds_{layer}.dat', dtype='float32', mode='w+', shape=layer.shape)
        # dataset_mmep[:] = embed
        prob_for_layer[:] = kernel.predict_proba(
            embed, batch_size=1_024 * 8
        ).T

        print("Computing internal correlations for layer", layer)
        corrs = utils.pairwise_correlation_metric(
            prob_for_layer, prob_for_layer)
        if use_saved:
            fo = open(f, 'wb+')
            pickle.dump(corrs, fo)
            fo.close()
        all_corrs.append(corrs)
    return all_corrs


class Decomposer:
    correlation_scores: List[List[List[float]]]
    # TODO: there has to be a smarter K-search alg
    # _k_search = N_DIMS * 2 #TODO: Back
    _k_search = 7
    # TODO: in params?
    # TODO: 2 params... one for lattice and one for scores
    # _weight_decay = 0.9
    _weight_decay_path_sel = 1.0  # TODO: should we go back to 1 here?
    # TODO: THERE IS STILL A PROBLEM WHERE WE REALLY AREN'T LOOKING AT TOTAL CORRELATION THROUGH THINGS...

    def __init__(self, params: paramslib.InterpParams,
                 device=DEFAULT_DEVICE,
                 n_max_features_per_neuron=6):
        torch.manual_seed(params.seed)
        np.random.seed(params.seed)
        self.device = device

        ds = get_dataset(params.dataset_name)
        model = load_model(params.model_name, device=device,
                              quantization=params.quantization)
        self.model = model
        shuffled = ds.shuffle(seed=params.seed)[
            'train'][:params.n_datasize]['text']
        dataset = shuffled
        layers = [i for i in range(params.n_blocks)]
        self.params = params
        print(
            f"Creating decomposer with parameter data hash {params.get_and_prepare_data_save_tag('start')}")
        print(
            f"Creating decomposer with parameter lattice hash {params.get_and_prepare_correlation_save_tag('start')}")
        # TODO: cutoff in random position

        if params.string_size_cutoff > 0:
            self.dataset = [utils.get_random_cutoff(
                d, params.string_size_cutoff) for d in dataset]
        else:
            self.dataset = dataset
        print("Created dataset")
        self.layers = layers
        self.correlation_scores = None
        self.ds_emb = get_layers_emb_dataset(
            self.model, self.dataset, self.layers, params=self.params, use_save=True)
        self.n_features_per_neuron = n_max_features_per_neuron
        print("Got embeddings")

    def load(self):
        self.correlation_scores = cluster_model_lattice(
            self.ds_emb, params=self.params)
        self.internal_correlations = correlate_internal_to_layer(
            self.ds_emb, params=self.params
        )

    def _get_ds_cache(self, ds: List[str], embeds: npt.NDArray = None):
        if embeds is None:
            embeds = get_layers_emb_dataset(
                self.model, ds, self.layers, params=self.params, use_save=False)
        token_to_original_ds = []
        token_to_pos_original_ds = []

        for i, d in enumerate(ds):
            tokenized = self.model[1](d)["input_ids"]
            for j in range(len(tokenized)):
                token_to_original_ds.append(i)
                token_to_pos_original_ds.append(j)

        assert len(token_to_original_ds) == len(
            token_to_pos_original_ds) == embeds[0].shape[0]
        return embeds, token_to_original_ds, token_to_pos_original_ds

    def score(self, to_score: List[str], layer: int, score_path: List[int], embeds: Union[npt.NDArray, None] = None, weighting_per_layer=None, use_log_scores=True) -> List[List[float]]:
        """
        Get the scores for the tokens in the dataset
        """
        if weighting_per_layer is None:
            weighting_per_layer = np.ones(self.params.n_blocks)

        embeds, token_to_original_ds, _ = self._get_ds_cache(
            to_score, embeds)
        all_scores = log_score_tokens_for_path(
            embd_dataset=embeds, path=score_path,
            score_weighting_per_layer=weighting_per_layer, layer=layer,
            ignore_weights=True)
        item_to_scores = {}
        # return log_scores
        # TODO: BATCHING!
        for i in range(len(all_scores)):
            item = token_to_original_ds[i]
            if item not in item_to_scores:
                item_to_scores[item] = []
            item_to_scores[item].append(all_scores[i])
        # Assuming that the scores are "dense" in how they were added, we have a list
        ks = sorted(list(item_to_scores.keys()))
        final_scores = []
        for k in ks:
            s = item_to_scores[k]
            if not use_log_scores:
                s = np.exp(s)
            final_scores.append(s)
        return final_scores

    def get_top_scores(self, dataset: List[str], path: List[int], layer: int, weighting_per_layer, embds=None, top_n=100):
        return get_top_scores(self, dataset, path, layer,
                              weighting_per_layer, embds, top_n)

    def scores_for_neuron(self, layer: int, neuron: int, dataset: List[str] = None, n_features_per_neuron=None, embds=None):
        if n_features_per_neuron is None:
            n_features_per_neuron = self.n_features_per_neuron
        if dataset is None:
            dataset = self.dataset
            embds = self.ds_emb

        n_blocks = len(self.layers)

        weighting_per_edge = np.ones(n_blocks - 1)
        for i in range(n_blocks - 1):
            if i < layer:
                weighting_per_edge[i] = self._weight_decay_path_sel ** (
                    layer - i - 1)
            else:
                weighting_per_edge[i] = self._weight_decay_path_sel ** (
                    i - layer)

        # weighting_per_layer_path_sel = utils.get_weighting_for_layer(
        #     layer, n_blocks, weight_decay=self._weight_decay_path_sel)
        # weighting_per_edge = np.concatenate((weighting_per_layer_path_sel[:layer],
        #                                     weighting_per_layer_path_sel[layer + 1:]))

        # We always "start" from the current layer"
        weighting_per_layer = utils.get_weighting_for_layer(
            layer, n_blocks)
        paths = graph.get_feature_paths(self.correlation_scores, layer, neuron,
                                        k_search=self._k_search, n_max_features=n_features_per_neuron,
                                        weighting_per_edge=weighting_per_edge, all_disjoint=True)
        scores_for_path = []
        print(f"Paths for neuron {neuron}", paths)
        for i, (path, _path_score) in enumerate(paths):
            toks, scores = self.get_top_scores(
                dataset, path, layer, weighting_per_layer, embds)
            scores_for_path.append((toks, scores))

            # TODO: maybe save json instead?
        visualization.save_display_for_neuron(
            scores_for_path, paths, layer, neuron)
        json_save_path = self.params.get_feature_json_path(layer, neuron)
        f = open(json_save_path, 'w+')
        json.dump(scores_for_path, f)
        f.close()
        print("Finished for neuron", layer, neuron)

    def scores_for_layer(self, layer: int, dataset: List[str] = None, embds=None, n_features_per_neuron=None, check_save=True):
        # TODO: meta tag
        # TODO: diff dem by feature
        for neuron in range(self.params.model_n_dims * 2):
            if check_save and os.path.exists(self.params.get_feature_json_path(layer, neuron)):
                print(
                    f"Already finished for layer {layer} and neuron {neuron}")
                continue
            else:
                self.scores_for_neuron(
                    layer, neuron, dataset, embds=embds, n_features_per_neuron=n_features_per_neuron)

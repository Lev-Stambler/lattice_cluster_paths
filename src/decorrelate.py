import os
import pickle
from datasets import Dataset, load_dataset
from typing import List, Union, Dict
import numpy as np
import numpy.typing as npt
import torch
import src.kernel as kernel
import src.params as paramslib
from src.model import TransformerModel, forward_hooked_model, load_model
import src.utils as utils
import src.visualization as visualization
import json
import src.graph as graph
import networkx as nx
from src.params import InterpParams, LatticeParams

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def _per_token_score_face_paths(embd_dataset: List[npt.NDArray], path: List[graph.Clique],
                                layer: int, layer_start=-1, BS=1_024 * 64):

    if layer_start == -1:
        layer_start = layer
    n_tokens = embd_dataset[0].shape[0]
    rets = np.ones(n_tokens)

    for i in range(0, n_tokens, BS):
        top_idx = min(i+BS, n_tokens)
        for curr_layer in range(layer_start, layer_start + len(path)):
            per_node_score = None
            n_nonzero_on = np.zeros(top_idx - i)
            curr_len = len(path[curr_layer - layer_start][1])
            for node in path[curr_layer - layer_start][1]:
                # TODO: func out because messy
                # multiplier = -1 if node % 2 == 1 else 1
                # selector = node // 2
                local_scores = kernel.feature_prob(
                    embd_dataset[curr_layer][i:top_idx], node, keep_negative=True)
                n_nonzero_on += (local_scores > 0)
                # TODO: get something like a cutoff!
                per_node_score = local_scores if per_node_score is None else per_node_score + local_scores

            # TODO: ghetto for nowj
            rets[i:top_idx] *= (per_node_score * (n_nonzero_on > len(path[curr_layer - layer_start][1]) / 3) if layer == curr_layer
                                else (n_nonzero_on >= (curr_len)))
    return rets

def get_layers_emb_dataset(model: TransformerModel, dataset: Dataset, layers: List[int], params: paramslib.InterpParams, use_save=True) -> List[npt.NDArray]:
    """
        Get the activations at every layer for a given dataset. We will flatten all the activations of a specific string to just get token specific activations
    """

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

# TODO: this is no longer log?


def score_tokens_for_path(embd_dataset: List[npt.NDArray],
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


def correlate_internal_layer(embeds: List[npt.NDArray], params: InterpParams, use_saved=True):
    """
        Get the "correlation" of activations on the standard basis
    """
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
    _k_search = 7

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
        # We count the activations *prior* to the first block (embeddings)
        # as being a "layer" and so have n_blocks + 1 embeddings
        # TODO: change back w/ later fix
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
        self.n_features_per_neuron = n_max_features_per_neuron
        print("Got embeddings")

    def load(self, reload=False):
        """
        Load embeddings, internal correlations, and other expensive objects.

        reload -- If true, parameters will be recomputed (equivalent to cleaning the cache directory and re-running load)
        """
        self.ds_emb = get_layers_emb_dataset(
            self.model, self.dataset, self.layers, params=self.params, use_save=not reload)
        self.internal_correlations = correlate_internal_layer(
            self.ds_emb, params=self.params, use_saved=not reload
        )

    def _get_dataset_cache(self, ds: List[str], embeds: npt.NDArray = None):
        """
            A wrapper to get the embeddings for a dataset.

            # TODO: have this be a bit nicer where we can cache if the dataset is the same and then reload
        """
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

    def tune_clique(self, clique: graph.Clique, layer: int) -> List[float]:
        """
            Given a clique, we will "tune" the activation parameters to maximize mutual signaling

            `returns` floats for each node in the clique corresponding to a minimum activation
        """
        embeds, _, _ = self._get_dataset_cache(
            self.dataset, self.ds_emb)
        embeds = embeds[layer]
        # TODO: I think that this should be cached
        embeds = utils.separate_neg_pos(embeds)
        print("AAHAHA", embeds.shape)
        clique_idxs = clique[1]
        maximizers = []
        for node in clique_idxs:
            a = embeds[:, node]
            B = embeds[:, clique_idxs]
            B = B.T
            print("FFFF", B.shape)
            maximizers.append(utils.signaling_maximization(a, B))
        return maximizers

        # 
        raise NotImplementedError
        pass

    def score_face_paths(self, clique_path: List[graph.Clique], layer: int, start_layer: int = -1):
        """
            Get the scores for the tokens in the dataset
        """
        embeds, token_to_original_ds, _ = self._get_dataset_cache(
            self.dataset, self.ds_emb)
        ret = _per_token_score_face_paths(
            self.ds_emb, clique_path, layer=layer, layer_start=start_layer)
        item_to_scores = {}

        # TODO: BATCHING!
        for i in range(len(ret)):
            item = token_to_original_ds[i]
            if item not in item_to_scores:
                item_to_scores[item] = []
            item_to_scores[item].append(ret[i])
        # Assuming that the scores are "dense" in how they were added, we have a list
        ks = sorted(list(item_to_scores.keys()))
        final_scores = []
        for k in ks:
            s = item_to_scores[k]
            final_scores.append(s)

        BOS_TOKEN = '||BOS||'
        scores_per_token_set = np.array([max(s) for s in final_scores])
        top_args = np.argsort(scores_per_token_set)[::-1][:100]
        # TODO: BOS?
        tokens = [[BOS_TOKEN] + [self.model[1].decode(t) for t in self.model[1](d)[
            'input_ids']] for d in self.dataset]
        tokens_reord = [tokens[i] for i in top_args]

        scores_reord = [final_scores[i] for i in top_args]
        return tokens_reord, scores_reord

    def scores_for_neuron(self, layer: int, neuron: int, dataset: List[str] = None, n_features_per_neuron=None, embds=None, degree_upperbound=1_000):
        if n_features_per_neuron is None:
            n_features_per_neuron = self.n_features_per_neuron
        if dataset is None:
            dataset = self.dataset
            embds = self.ds_emb

        G = graph.graph_from_correlations(self.internal_correlations[layer])
        G = graph.sparsify_weighted_graph(G, degree_upperbound)
        weight_attrs = nx.get_edge_attributes(G, 'weight')
        clique_iterator = nx.find_cliques(G, nodes=[neuron])
        cliques = []
        for c in clique_iterator:
            cliques.append((graph.get_clique_score(weight_attrs, c), c))
            if len(cliques) > self.n_features_per_neuron:
                break
        scores_for_path = []
        print("GOT CLIQUES WITH LENGTH", [len(c[1]) for c in cliques], cliques)
        paths = []
        # We always "start" from the current layer"
        for i, clique in enumerate(cliques):
            toks, scores = self.score_face_paths(
                [clique], layer)
            paths.append([clique])
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

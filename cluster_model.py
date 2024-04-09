import os
import pickle
from datasets import Dataset, load_dataset
from typing import List, Union, Dict
import numpy as np
import numpy.typing as npt
import torch
import transformer_lens
import kernel
import metadata as paramslib
import utils
import visualization
import graph
import json

from metadata import InterpParams, LatticeParams

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_top_scores(self, dataset: List[str], path: List[int],
                   layer: int, weighting_per_layer, embds=None, top_n=100):
    model = self.model_lens
    n_blocks = len(path)
    BOS_TOKEN = '||BOS||'
    score_path = path
    print(weighting_per_layer)
    scores = self.score(
        dataset,
        layer=layer,
        score_path=score_path,
        weighting_per_layer=weighting_per_layer,
        use_log_scores=True,
        embeds=embds
    )

    scores_per_token_set = np.array([max(s) for s in scores])
    top_args = np.argsort(scores_per_token_set)[::-1]
    # TODO: BOS?
    tokens = [[BOS_TOKEN] + [model.tokenizer.decode(t) for t in model.tokenizer(d)[
        'input_ids']] for d in dataset]
    tokens_reord = [tokens[i] for i in top_args]
    scores_reord = [scores[i] for i in top_args]
    return tokens_reord[:top_n], scores_reord[:top_n]

# TODO: by different model parameterize?


def get_block_base_label(i): return f'blocks.{i}'


def get_block_out_label(i): return f'{get_block_base_label(i)}.hook_resid_post'


def forward_on_block(model, block_idx: int, data: npt.NDArray):
    ret = model.blocks[block_idx](torch.tensor(list(data)).unsqueeze(
        0).unsqueeze(0).to(device=DEFAULT_DEVICE)).detach().cpu().numpy()
    return ret[0][0]


def forward_pass(model_lens: transformer_lens.HookedTransformer, t: str, layer: str) -> torch.Tensor:
    with torch.no_grad():
        return model_lens.run_with_cache(t)[1][layer]


def get_layers_emb_dataset(model_lens: transformer_lens.HookedTransformer, dataset: Dataset, layers: List[str], params: paramslib.InterpParams, use_save=True) -> List[npt.NDArray]:

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
            # print("ON", t)
            top_idx = min(t + BS, len(dataset))
            d = dataset[t:top_idx]
            # torch.cuda.empty_cache()
            outs = model_lens.run_with_cache(d)[1]
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
        # BS = 1_024 * 8
        BS = 8
        for i in range(0, total_n_toks, BS):
            # if i % 200 == 0:
            #     print("Numpy ON", i)
            top_idx = min(i + BS, total_n_toks)
            # print("ON", i, top_idx, all_outs[l][i:top_idx])
            out_np[i:top_idx, :] = np.array(all_outs[l][i:top_idx])
        outs_np.append(out_np)
    print("Finished and saving to file")
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
            ds_mmep, batch_size=128).T
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


# TODO: this is no longer log?
def log_score_tokens_for_path(embd_dataset: List[npt.NDArray],
                              path: List[int],
                              score_weighting_per_layer: npt.NDArray, BS=1_024*64):
    """
    embd_dataset: The outer index corresponds to the layer, the inner index corresponds to the token
    """
    # Set the outer index to the token
    # token_per_layer = embd_dataset.swapaxes(0, 1)
    n_tokens = embd_dataset[0].shape[0]
    scores = np.ones(n_tokens)
    assert len(embd_dataset) == len(path)

    for i in range(0, n_tokens, BS):
        # if i % (BS * 10) == 0:
        #     print("Scoring on", i, "of", n_tokens)
        top_idx = min(i + BS, n_tokens)
        for layer in range(len(path)):
            local_scores = kernel.feature_prob(
                embd_dataset[layer][i:top_idx], path[layer])

            # Make sure that we never multiply by a negative number
            # Negative just means that we are anti-correlated
            # TODO: JUST 0 / 1?
            local_scores = local_scores * (local_scores > 0.0)
            scores[i:top_idx] *= local_scores ** score_weighting_per_layer[layer]
            # local_scores
    scores = scores  # Make sure that we are always positive
    return scores


def get_transformer(name: str, quantization: str = None, device=DEFAULT_DEVICE):
    """
    Get the transformer model from the name
    """
    # tokenizer = AutoTokenizer.from_pretrained(name)
    if quantization is None:
        model = transformer_lens.HookedTransformer.from_pretrained(name).to(device)
    elif quantization == '8bit':
        model = transformer_lens.HookedTransformer.from_pretrained(name, load_in_8bit=True).to(device)
    elif quantization == '4bit':
        model = transformer_lens.HookedTransformer.from_pretrained(name, load_in_4bit=True).to(device)
    else:
        raise ValueError("Expected either no quantization or 8bit or 4bit")
    tokenizer = model.tokenizer
    return model, tokenizer


def get_dataset(name: str):
    """
    Get the dataset from the name
    """
    return load_dataset(name)


class Decomposer:
    correlation_scores: List[List[List[float]]]
    # TODO: there has to be a smarter K-search alg
    # _k_search = N_DIMS * 2 #TODO: Back
    _k_search = 20
    # TODO: in params?
    # TODO: 2 params... one for lattice and one for scores
    # _weight_decay = 0.9
    _weight_decay_path_sel = 0.80  # TODO: should we go back to 1 here?
    # TODO: THERE IS STILL A PROBLEM WHERE WE REALLY AREN'T LOOKING AT TOTAL CORRELATION THROUGH THINGS...

    def __init__(self, params: paramslib.InterpParams,
                 n_max_features_per_neuron=4):
        torch.manual_seed(params.seed)
        np.random.seed(params.seed)

        ds = get_dataset(params.dataset_name)
        model, _ = get_transformer(params.model_name)
        self.model_lens = model
        shuffled = ds.shuffle(seed=params.seed)[
            'train'][:params.n_datasize]['text']
        dataset = shuffled
        layers = [get_block_out_label(i) for i in range(params.n_blocks)]
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
        self.labs = [get_block_out_label(i) for i in range(params.n_blocks)]
        self.ds_emb = get_layers_emb_dataset(
            self.model_lens, self.dataset, self.labs, params=self.params, use_save=True)
        self.n_features_per_neuron = n_max_features_per_neuron
        print("Got embeddings")

    def load(self):
        self.correlation_scores = cluster_model_lattice(
            self.ds_emb, params=self.params)

    def _get_ds_metadata(self, ds: List[str], embeds: npt.NDArray = None):
        if embeds is None:
            embeds = get_layers_emb_dataset(
                self.model_lens, ds, self.layers, params=self.params, use_save=False)
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
            token_to_pos_original_ds) == embeds[0].shape[0]
        return embeds, token_to_original_ds, token_to_pos_original_ds

    def score(self, to_score: List[str], layer: int, score_path: List[int], embeds: Union[npt.NDArray, None] = None, weighting_per_layer=None, use_log_scores=True) -> List[List[float]]:
        """
        Get the scores for the tokens in the dataset
        """
        if weighting_per_layer is None:
            weighting_per_layer = np.ones(self.params.n_blocks)

        embeds, token_to_original_ds, _ = self._get_ds_metadata(
            to_score, embeds)
        log_scores = log_score_tokens_for_path(
            embd_dataset=embeds, path=score_path,
            score_weighting_per_layer=weighting_per_layer)
        item_to_scores = {}
        # return log_scores
        # TODO: BATCHING!
        for i in range(len(log_scores)):
            item = token_to_original_ds[i]
            if item not in item_to_scores:
                item_to_scores[item] = []
            item_to_scores[item].append(log_scores[i])
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
                    layer - i)
            else:
                weighting_per_edge[i] = self._weight_decay_path_sel ** (
                    i - layer + 1)

        # weighting_per_layer_path_sel = utils.get_weighting_for_layer(
        #     layer, n_blocks, weight_decay=self._weight_decay_path_sel)
        # weighting_per_edge = np.concatenate((weighting_per_layer_path_sel[:layer],
        #                                     weighting_per_layer_path_sel[layer + 1:]))

        # We always "start" from the current layer"
        weighting_per_layer = utils.get_weighting_for_layer(
            layer, n_blocks)
        weighting_per_layer[layer] = 1.0
        print("WEIGHTING PER LAYER", weighting_per_layer)
        print("EDGE DISCOVERY WEIGHTING PER LAYER", weighting_per_edge)
        paths = graph.get_feature_paths(self.correlation_scores, layer, neuron,
                                        k_search=self._k_search, n_max_features=n_features_per_neuron,
                                        weighting_per_edge=weighting_per_edge)
        scores_for_path = []
        print("PATHS", paths)
        for i, (path, _path_score) in enumerate(paths):
            print("LOOKING AT PATH", path, i + 1, "out of", len(paths))
            t, s = self.get_top_scores(
                dataset, path, layer, weighting_per_layer, embds)
            scores_for_path.append((t, s))
            print("Done with path", i + 1, "out of", len(paths))

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

#     def scores_for_all(self, dataset: List[str] = None, embds=None):
#         # TODO: check cached!!
#         for layer in range(self.params.n_blocks):
#             self.scores_for_layer(layer, dataset=dataset, embds=embds)

# # TODO: MUTUAL INFORMATION!!!!

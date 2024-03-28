from typing import List
from circuitsvis.activations import text_neuron_activations
import torch
import numpy as np
from einops import rearrange
import networkx as nx
import heapq
import transformer_lens
# TODO: this is just a c and p rn

# Get the activations for the best dict features
# TODO: fix up


def get_feature_datapoints(feature_index, dictionary_activations, tokenizer, token_amount, dataset, k=10, setting="max"):
    best_feature_activations = dictionary_activations[:, feature_index]
    print("AAAAA", best_feature_activations.shape)
    # Sort the features by activation, get the indices
    if setting == "max":
        found_indices = torch.argsort(
            best_feature_activations, descending=True)[:k]
    elif setting == "uniform":
        # min_value = torch.min(best_feature_activations)
        min_value = torch.min(best_feature_activations)
        max_value = torch.max(best_feature_activations)

        # Define the number of bins
        num_bins = k

        # Calculate the bin boundaries as linear interpolation between min and max
        bin_boundaries = torch.linspace(min_value, max_value, num_bins + 1)

        # Assign each activation to its respective bin
        bins = torch.bucketize(best_feature_activations, bin_boundaries)

        # Initialize a list to store the sampled indices
        sampled_indices = []

        # Sample from each bin
        for bin_idx in torch.unique(bins):
            if (bin_idx == 0):  # Skip the first one. This is below the median
                continue
            # Get the indices corresponding to the current bin
            bin_indices = torch.nonzero(
                bins == bin_idx, as_tuple=False).squeeze(dim=1)

            # Randomly sample from the current bin
            sampled_indices.extend(np.random.choice(
                bin_indices, size=1, replace=False))

        # Convert the sampled indices to a PyTorch tensor & reverse order
        found_indices = torch.tensor(sampled_indices).long().flip(dims=[0])
    else:  # random
        # get nonzero indices
        nonzero_indices = torch.nonzero(best_feature_activations)[:, 0]
        # shuffle
        shuffled_indices = nonzero_indices[torch.randperm(
            nonzero_indices.shape[0])]
        found_indices = shuffled_indices[:k]
    num_datapoints = int(dictionary_activations.shape[0]/token_amount)
    datapoint_indices = [np.unravel_index(
        i, (num_datapoints, token_amount)) for i in found_indices]
    text_list = []
    full_text = []
    token_list = []
    full_token_list = []
    for md, s_ind in datapoint_indices:
        md = int(md)
        s_ind = int(s_ind)
        full_tok = torch.tensor(dataset[md]["input_ids"])
        full_text.append(tokenizer.decode(full_tok))
        tok = dataset[md]["input_ids"][:s_ind+1]
        text = tokenizer.decode(tok)
        text_list.append(text)
        token_list.append(tok)
        full_token_list.append(full_tok)
    return text_list, full_text, token_list, full_token_list


def get_neuron_activation(token, feature, model, autoencoder, layer, setting="dictionary_basis"):
    with torch.no_grad():
        _, cache = model.run_with_cache(token.to(model.cfg.device))
        cache_name = f"blocks.{layer}.hook_resid_post"
        neuron_act_batch = cache[cache_name]
        if setting == "dictionary_basis":
            act = autoencoder.encode(neuron_act_batch.squeeze(0))
            return act[:, feature].tolist()
        else:  # neuron/residual basis
            return neuron_act_batch[0, :, feature].tolist()


def ablate_text(text, feature, model, autoencoder, layer,  setting="plot"):
    if isinstance(text, str):
        text = [text]
    display_text_list = []
    activation_list = []
    for t in text:
        # Convert text into tokens
        if isinstance(t, str):  # If the text is a list of tokens
            split_text = model.to_str_tokens(t, prepend_bos=False)
            tokens = model.to_tokens(t, prepend_bos=False)
        else:  # t equals tokens
            tokens = t
            split_text = model.to_str_tokens(t, prepend_bos=False)
        seq_size = tokens.shape[1]
        if (seq_size == 1):  # If the text is a single token, we can't ablate it
            continue
        original = get_neuron_activation(
            tokens, feature, model, autoencoder, layer)[-1]
        changed_activations = torch.zeros(seq_size).cpu()
        for i in range(seq_size):
            # Remove the i'th token from the input
            ablated_tokens = torch.cat((tokens[:, :i], tokens[:, i+1:]), dim=1)
            changed_activations[i] += get_neuron_activation(
                ablated_tokens, feature, model, autoencoder, layer)[-1]
        changed_activations -= original
        display_text_list += [x.replace('\n', '\\newline')
                              for x in split_text] + ["\n"]
        activation_list += changed_activations.tolist() + [0.0]
    activation_list = torch.tensor(activation_list).reshape(-1, 1, 1)
    if setting == "plot":
        return text_neuron_activations(tokens=display_text_list, activations=activation_list)
    else:
        return display_text_list, activation_list


def visualize_text(model: transformer_lens.HookedTransformer, tokens: List[List[int]], scores_per_token: List[List[float]]):
    texts = [model.tokenizer.decode(ts) for ts in tokens]
    for text, ts, scores in zip(texts, tokens, scores_per_token):
        text_neuron_activations(tokens=text, activations=scores)


def ablate_feature_direction(tokens, feature, model, autoencoder, layer):
    def mlp_ablation_hook(value, hook):
        # Rearrange to fit autoencoder
        int_val = rearrange(value, 'b s h -> (b s) h')

        # Run through the autoencoder
        act = autoencoder.encode(int_val)
        feature_to_ablate = feature  # TODO: bring this out of the function

        dictionary_for_this_autoencoder = autoencoder.get_learned_dict()
        feature_direction = torch.outer(act[:, feature_to_ablate].squeeze(
        ), dictionary_for_this_autoencoder[feature_to_ablate].squeeze())
        batch, seq_len, hidden_size = value.shape
        feature_direction = rearrange(
            feature_direction, '(b s) h -> b s h', b=batch, s=seq_len)
        value -= feature_direction
        return value

    cache_name = f"blocks.{layer}.hook_resid_post"
    return model.run_with_hooks(tokens,
                                fwd_hooks=[(
                                    cache_name,
                                    mlp_ablation_hook
                                )]
                                )


def add_feature_direction(tokens, feature, model, autoencoder, scalar=1.0):
    def residual_add_hook(value, hook):
        feature_direction = autoencoder.decoder.weight[:, feature].squeeze()
        value += scalar*feature_direction
        return value

    return model.run_with_hooks(tokens,
                                fwd_hooks=[(
                                    cache_name,
                                    residual_add_hook
                                )]
                                )


def ablate_feature_direction_display(text, autoencoder, model, layer, features=None, setting="true_tokens", verbose=False):
    if isinstance(features, int):
        features = torch.tensor([features])
    if isinstance(features, list):
        features = torch.tensor(features)
    if isinstance(text, str):
        text = [text]
    text_list = []
    logit_list = []
    for t in text:
        tokens = model.to_tokens(t, prepend_bos=False)
        with torch.no_grad():
            original_logits = model(tokens).log_softmax(-1).cpu()
            ablated_logits = ablate_feature_direction(
                tokens, features, model, autoencoder, layer).log_softmax(-1).cpu()
        # ablated > original -> negative diff
        diff_logits = ablated_logits - original_logits
        tokens = tokens.cpu()
        if setting == "true_tokens":
            split_text = model.to_str_tokens(t, prepend_bos=False)
            # TODO: verify this is correct
            gather_tokens = rearrange(tokens[:, 1:], "b s -> b s 1")
            # Gather the logits for the true tokens
            diff = rearrange(
                diff_logits[:, :-1].gather(-1, gather_tokens), "b s n -> (b s n)")
        elif setting == "max":
            # Negate the diff_logits to see which tokens have the largest effect on the neuron
            val, ind = (-1*diff_logits).max(-1)
            diff = rearrange(val[:, :-1], "b s -> (b s)")
            diff *= -1  # Negate the values gathered
            split_text = model.to_str_tokens(ind, prepend_bos=False)
            gather_tokens = rearrange(ind[:, 1:], "1 s -> 1 s 1")
        # Remove the first token since we're not predicting it
        split_text = split_text[1:]
        if (verbose):
            text_list += [x.replace('\n', '\\newline')
                          for x in split_text] + ["\n"]
            text_list += [x.replace('\n', '\\newline')
                          for x in split_text] + ["\n"]
            orig = rearrange(
                original_logits[:, :-1].gather(-1, gather_tokens), "b s n -> (b s n)")
            ablated = rearrange(
                ablated_logits[:, :-1].gather(-1, gather_tokens), "b s n -> (b s n)")
            logit_list += orig.tolist() + [0.0]
            logit_list += ablated.tolist() + [0.0]
        text_list += [x.replace('\n', '\\newline')
                      for x in split_text] + ["\n"]
        logit_list += diff.tolist() + [0.0]
    logit_list = torch.tensor(logit_list).reshape(-1, 1, 1)
    if verbose:
        print(
            f"Max & Min logit-diff: {logit_list.max().item():.2f} & {logit_list.min().item():.2f}")
    return text_neuron_activations(tokens=text_list, activations=logit_list)


def generate_text(input_text, num_tokens, model, autoencoder, feature, temperature=0.7, setting="add", scalar=1.0):
    # Convert input text to tokens
    input_ids = model.tokenizer.encode(
        input_text, return_tensors='pt').to(autoencoder.encoder.device)

    for _ in range(num_tokens):
        # Generate logits
        with torch.no_grad():
            if (setting == "add"):
                logits = add_feature_direction(
                    input_ids, feature, model, autoencoder, scalar=scalar)
            else:
                logits = model(input_ids)

        # Apply temperature
        logits = logits / temperature

        # Sample from the distribution
        probs = torch.nn.functional.softmax(logits[:, -1, :], dim=-1)
        predicted_token = torch.multinomial(probs, num_samples=1)

        # Append predicted token to input_ids
        input_ids = torch.cat((input_ids, predicted_token), dim=-1)

    # Decode the tokens to text
    output_text = model.tokenizer.decode(input_ids[0])

    return output_text

# Logit Lens


def logit_lens(model, best_feature, dictionary):
    with torch.no_grad():
        # There are never-used tokens, which have high norm. We want to ignore these.
        bad_ind = (model.W_U.norm(dim=0) > 20)
        feature_direction = dictionary[best_feature].to(model.cfg.device)
        # feature_direction = torch.matmul(feature_direction, model.W_out[layer]) # if MLP
        logits = torch.matmul(feature_direction, model.W_U).cpu()
    # Don't include bad indices
    logits[bad_ind] = -1000
    topk_values, topk_indices = torch.topk(logits, 20)
    top_text = model.to_str_tokens(topk_indices)
    print(f"{top_text}")
    print(topk_values)


def top_k_dag_paths_dynamic(layers: List[List[List[float]]], start_layer, top_layer, k, weight='weight'):
    assert len(layers) > 1, "Need at least 2 layers"
    assert start_layer < top_layer, "Start layer must be less than top layer"
    assert top_layer < len(layers), "Top layer must be less than the number of layers"
    
    memoizer = {}

    def recur(layer: int, node_layer_idx: int, top_k_per_layer):
        if (layer, node_layer_idx) in memoizer:
            return memoizer[(layer, node_layer_idx)]

        # Base case
        if layer == 0:
            return top_k_per_layer[layer]

        past_layer = layer - 1
        past_layer_vals = layers[past_layer]

        # Get the inbound values
        vs = [s[node_layer_idx] for s in past_layer_vals]

        new_top_k = []
        for i, v in enumerate(vs):
            for (node_path, val) in top_k_per_layer[layer]:
                new_top_k.append(([i] + [node_path], v + val))

        new_top_k.sort(key=lambda x: x[1], reverse=True)[:k]

        all_rets = []
        for past_layer_idx in range(len(past_layer_vals)):
            ret = recur(past_layer, past_layer_idx, new_top_k)
            memoizer[(layer, node_layer_idx)] = ret
            all_rets.append(ret)
        all_flat = [item for sublist in all_rets for item in sublist]
        top_rets = all_flat.sort(key=lambda x: x[1], reverse=True)[:k]

        return top_rets
        # memoizer[(layer, node)] = 
    # TODO: return

# We have to implement https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3009499/


def top_k_paths_to_end(G, start, end, k, weight='weight'):
    # Perform topological sort on the DAG
    topo_sort = list(nx.topological_sort(G))

    # Initialize dictionaries for storing the maximum path weights and paths
    max_weight = {node: float('-inf') for node in G}
    max_weight[start] = 0
    paths = {node: [] for node in G}
    paths[start] = [[start]]

    # Heap to maintain top k paths
    top_paths = []

    # Update weights and paths based on the topological order
    for node in topo_sort:
        for successor in G.successors(node):
            edge_weight = G[node][successor][weight]
            new_weight = max_weight[node] + edge_weight

            # Update only if the new weight is better
            if new_weight > max_weight[successor]:
                max_weight[successor] = new_weight
                paths[successor] = [path + [successor] for path in paths[node]]
            # Handle paths with equal weight
            elif new_weight == max_weight[successor]:
                paths[successor].extend([path + [successor]
                                        for path in paths[node]])

    # Only consider paths that end at the 'end' vertex
    for path in paths[end]:
        path_weight = sum(G[path[i]][path[i + 1]][weight]
                          for i in range(len(path) - 1))
        if len(top_paths) < k:
            heapq.heappush(top_paths, (path_weight, path))
        else:
            heapq.heappushpop(top_paths, (path_weight, path))

    # Sort the results based on weights in descending order before returning
    top_paths.sort(reverse=True, key=lambda x: x[0])
    return top_paths

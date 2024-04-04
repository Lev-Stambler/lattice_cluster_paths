#!/usr/bin/env python
# coding: utf-8

# In[1]:


from cluster_model import get_dataset, get_transformer, DATASET_NAME, MODEL_NAME, SEED, N_DATASIZE, get_block_out_label, N_BLOCKS
import cluster_model

ds = get_dataset(DATASET_NAME)
model, _ = get_transformer(MODEL_NAME)
shuffled = ds.shuffle(seed=SEED)['train'][:N_DATASIZE]['text']

ds = shuffled
labs = [get_block_out_label(i) for i in range(N_BLOCKS)]


# In[2]:


import importlib
importlib.reload(cluster_model)

# TODO: auto sim cutoff needed
decomp = cluster_model.Decomposer(model, ds, labs)
decomp.load()


# In[ ]:


decomp.lattice_scores[0].shape


# 

# In[ ]:


from typing import List
import utils
importlib.reload(utils)
modified_lattice = decomp.lattice_scores.copy()

# TODO: have function which does something like fixing from the a specific neuron?
# Ie have the sub-lattice with only **related** neurons

def cutoff_lattice(lattice: List[List[List[float]]], related_cutoff = 1):
    print(lattice[0].sum())
    r = [(layer > related_cutoff) * layer for layer in lattice]
    print(r[0].sum())
    return r

def create_sublattice(lattice, layer: int, idx_in_layer: int, related_cutoff = 10):
	with_cutoff = cutoff_lattice(lattice, related_cutoff)

LAST_FEAT = 20
modified_lattice = cutoff_lattice(decomp.lattice_scores, related_cutoff=15)

# utils.top_k_dag_paths_dynamic(modified_lattice, k=10_000)


# ## Isolate Specific Neurons

# In[ ]:


import numpy as np

def get_weighting_for_layer(layer: int, n_layers: int, weight_decay=0.95):
    r = np.ones(n_layers)
    r[layer] = 1
    G = weight_decay
    for i in range(layer):
        r[i] = G ** (layer - i)
    for i in range(layer + 1, n_layers):
        # Decrease by Gx per layer
        r[i] = G ** (i - layer)
    return r
    


# In[ ]:


decomp.lattice_scores[0][:, 13]


# In[ ]:


importlib.reload(utils)
# TODO: restrict lattice, change picture instead of starting elsewhere...
# ~~Use source~~ node
# TODO: ~~SOMEHOW LAYER 2 is where stuff happens~~
cutoff = 0.3
scores_cutoff = [m * m > cutoff for m in modified_lattice]
paths = utils.top_k_dag_paths(decomp.lattice_scores, layer=1, neuron=13, k=50)
paths


# ## Setup for scoring and visualization

# In[ ]:


import math
import numpy as np
from IPython.core.display import display, HTML
from circuitsvis.utils.render import render


# ## Try different scoring method

# In[ ]:


# # TODO: scoring and testing the paths

# importlib.reload(cluster_model)

# # score_path = [69, 2, 493, 289, 511, 48]
# # TODO: embeds to here...
# to_score = ds
# to_score_adj = [s[:300] for s in to_score][:300]
# to_score = to_score_adj
# importlib.reload(cluster_model)
# # TODO: should we work over log everywhere?
# _, top_ds, scores = cluster_model.score_for_neuron(decomp, to_score_adj, LAYER, score_path, top_n=200, weighting_per_layer=get_weighting_for_layer(LAYER, N_BLOCKS))

# def get_renderable_scores(scored_t: List[str], scores: List


# scores_per_token_set = np.array([max(s) for s in scores])
# top_args = np.argsort(scores_per_token_set)[::-1]

# # scores_per_token_set = [max(s) for s in scores]
# # top_args = [s.argmax() for s in scores]

# to_score_top = [to_score[i] for i in top_ds]

# tokens = [[model.tokenizer.decode(t) for t in model.tokenizer(d)[
#     'input_ids']] for d in to_score_top]
# tokens_reord = [tokens[i] for i in top_args]
# scores_reord = [scores[i] for i in top_args]

# # TODO: WHAT IS HAPPENING WITH NAN?
# act_simp = [[[[math.exp(10 * tok)]]
#              for tok in s] for s in scores_reord]
# # TODO sep fun
# html = render(
#     "TextNeuronActivations",
#     tokens=tokens_reord,
#     activations=act_simp,
#     firstDimensionName="Layer",
#     secondDimensionName="Neuron",
#     firstDimensionLabels=None,
#     secondDimensionLabels=None
# )
# display(HTML(str(html)))


# ## Try more path like method

# In[ ]:


"""
score_path = [13, 13, 391, 57, 57, 57]... relates to laws/ licensing
"""

# TODO: SIOMETHING WRONG HERE WITH THE PATHS
import numpy as np
import metric
importlib.reload(cluster_model)
importlib.reload(utils)
importlib.reload(metric)
# score_path = [70, 13, 59, 423, 368, 418]
score_path = [105, 13, 460, 112, 87, 402] # Opening " in code block thing?
LAYER = 1
# score_path = [0, 0, 69, 51, 195, 99]
# score_path = [69, 1, 107, 289, 97, 508]

"""
Paths which seem to have meaning:

"""

weighting_per_layer = get_weighting_for_layer(LAYER, N_BLOCKS, weight_decay=0.9)
# weighting_per_layer[0] = 0
# weighting_per_layer[1] = 0
print(weighting_per_layer)
to_score = [d[:300] for d in ds][:200]
scores = decomp.score(
    to_score,
    score_path=score_path,
    # TODO: how do we weight???
    weighting_per_layer=weighting_per_layer,
    use_log_scores=True
)

scores_per_token_set = np.array([max(s) for s in scores])
top_args = np.argsort(scores_per_token_set)[::-1]

# TODO: we have the weights here

tokens = [[model.tokenizer.decode(t) for t in model.tokenizer(d)[
    'input_ids']] for d in to_score]
tokens_reord = [tokens[i] for i in top_args]
scores_reord = [scores[i] for i in top_args]


# In[ ]:


# TODO: what up with this max
max_score = max(max(s for s in scores_reord))
max_score


# In[ ]:


import math
from IPython.core.display import display, HTML
import math
import numpy as np

act_simp = [[[[math.exp(tok - max_score)]]
             for tok in s] for s in scores_reord]
# TODO sep fun
html = render(
    "TextNeuronActivations",
    tokens=tokens_reord,
    activations=act_simp,
    firstDimensionName="Layer",
    secondDimensionName="Neuron",
    firstDimensionLabels=None,
    secondDimensionLabels=None
)
display(HTML(str(html)))


# ## Look at distributions of scores

# In[ ]:


import matplotlib.pyplot as plt
import seaborn as sns

# Get distribution of lattice scores

LAYER_START = 4
cutoff = 0.1

# ax = sns.heatmap(decomp.lattice_scores[LAYER_START])
plt.imshow((decomp.lattice_scores[LAYER_START] > cutoff), cmap='hot', interpolation='nearest')
plt.show()

plt.plot((decomp.lattice_scores[3] > cutoff).sum(axis=1))


# In[ ]:





# 

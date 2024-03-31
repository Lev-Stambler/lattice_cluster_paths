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
decomp = cluster_model.Decomposer(model, ds, labs, similarity_cutoff=2)
decomp.load()


# In[3]:


decomp.lattice_scores[0].shape


# In[4]:


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


# In[5]:


# TODO:
""""
We need to combine the 
"""

import numpy as np
from utils import get_random_cutoff

score_path = [8, 57, 89]
score_path = [10, 63, 89]
score_path = [3, 3, 1]
# score_path = [39, 14, 2, 3, 4]
score_path = [1, 12, 2, 3, 4, 5]
"""
Paths which seem to have meaning

score_path = [1, 1, 2, 3, 4]
Very strong for Q: . Specifically on the space
"""

to_score = [get_random_cutoff(d, 100) for d in ds][:300]
scores = decomp.score(
    to_score,
    score_path=score_path,
    weighting_per_layer=np.array([1, 1, 1, 1, 1, 1])
)


# In[6]:


min([min(s) for s in scores]), max([max(s) for s in scores])


# In[7]:


from IPython.core.display import display, HTML
import math
import numpy as np

scores_per_token_set = np.array([max(s) for s in scores])
top_args = np.argsort(scores_per_token_set)[::-1]

tokens = [[model.tokenizer.decode(t) for t in model.tokenizer(d)[
    'input_ids']] for d in to_score]
tokens_reord = [tokens[i] for i in top_args]
scores_reord = [scores[i] for i in top_args]
# TODO: SORT!
act_simp = scores_reord
act_simp = [[[[0 if math.isnan(tok * 100) else int(math.exp(tok * 0.001))]] for tok in s] for s in scores_reord]
# TODO sep fun
from circuitsvis.utils.render import RenderedHTML, render
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


# ## Isolate Specific Neurons

# In[8]:


avoid_set = {
	0: [42],
	1: [55],
}


# In[9]:


import os
import cluster_model
importlib.reload(cluster_model)

LAYER = 3
NODE = 10
# TODO: this ain't right!! We need specific paths!!
ret = cluster_model.restrict_to_related_path(decomp.lattice_scores, LAYER, NODE,
											avoid_set=avoid_set,
                                              n_paths=10_000)
# Write to file
with open("/tmp/related_paths.txt", "w") as f:
    f.write("\n".join([str(r) for r in ret]))
len(ret)


# ## Try different scoring method

# In[10]:


# importlib.reload(cluster_model)

# score_path = [0, 16, 10, 10, 31, 8]
# # TODO: embeds to here...
# importlib.reload(cluster_model)
# _, top_ds, scores = cluster_model.score_for_neuron(decomp, to_score[:200], LAYER, score_path, top_n=20, weighting_per_layer=[0, 1, 1, 1, 1, 1])


# ## Try more path like method

# In[20]:


importlib.reload(cluster_model)
score_path = [83, 6, 47, 10, 48, 12]

"""
Paths which seem to have meaning

score_path = [1, 1, 2, 3, 4]
Very strong for Q: . Specifically on the space

score_path_path = [3, 15, 19, 11, 12]
Very strong for Q: . when context is about *programming*
"""

weighting_per_layer = np.ones(len(decomp.lattice_scores) + 1)
# TODO: do we need to weight the layers?
weighting_per_layer[LAYER] = 1.03

# weighting_per_layer[0] = 1
# weighting_per_layer[1] = 1

to_score = [d[:200] for d in ds][:600]
scores = decomp.score(
    to_score,
    score_path=score_path,
    # TODO: how do we weight???
    weighting_per_layer=weighting_per_layer
)

scores_per_token_set = np.array([max(s) for s in scores])
top_args = np.argsort(scores_per_token_set)[::-1]

tokens = [[model.tokenizer.decode(t) for t in model.tokenizer(d)[
    'input_ids']] for d in to_score]
tokens_reord = [tokens[i] for i in top_args]


# In[21]:


scores_min = min([min(s) for s in scores_reord])
# scores_rebal = [[(s - scores_min) for s in score] for score in scores_reord]


# In[22]:


# TODO: WHAT IS HAPPENING WITH NAN?
act_simp = [[[[int(math.exp(tok * 0.001))]]
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


# 

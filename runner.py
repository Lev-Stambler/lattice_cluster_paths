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

score_path = [8, 57, 89]
score_path = [10, 63, 89]
score_path = [3, 3, 1]
# score_path = [39, 14, 2, 3, 4]
score_path = [1, 1, 2, 3, 4]
"""
Paths which seem to have meaning

score_path = [1, 1, 2, 3, 4]
Very strong for Q: . Specifically on the space
"""

to_score = [d[:100] for d in ds][:300]
scores = decomp.score(
    to_score,
    score_path=score_path,
    weighting_per_layer=np.array([1, 1, 1, 1, 1])
)


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
act_simp = [[[[0 if math.isnan(tok * 100) else int(math.exp(tok * 100))]] for tok in s] for s in scores_reord]
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

# In[99]:


import cluster_model
importlib.reload(cluster_model)

LAYER = 0
NODE = 3
ret = cluster_model.restrict_to_related_vertex(decomp.lattice_scores, LAYER, NODE, rel_cutoff=13.1)
len(ret), ret


# In[105]:


score_path_path = [3, 15, 19, 11, 30]

"""
Paths which seem to have meaning

score_path = [1, 1, 2, 3, 4]
Very strong for Q: . Specifically on the space

score_path_path = [3, 15, 19, 11, 12]
Very strong for Q: . when context is about *programming*
"""

to_score = [d[:100] for d in ds][:300]
scores = decomp.score(
    to_score,
    score_path=score_path,
    weighting_per_layer=np.array([1, 1, 1, 1, 1]) # TODO: how do we weight???
)

scores_per_token_set = np.array([max(s) for s in scores])
top_args = np.argsort(scores_per_token_set)[::-1]

tokens = [[model.tokenizer.decode(t) for t in model.tokenizer(d)[
    'input_ids']] for d in to_score]
tokens_reord = [tokens[i] for i in top_args]
scores_reord = [scores[i] for i in top_args]
# TODO: SORT!
act_simp = scores_reord
act_simp = [[[[0 if math.isnan(tok * 100) else int(math.exp(tok * 100))]] for tok in s] for s in scores_reord]
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

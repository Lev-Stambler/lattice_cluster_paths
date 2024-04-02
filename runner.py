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


[d.min() for d in decomp.lattice_scores]
[d.max() for d in decomp.lattice_scores]


# ## Isolate Specific Neurons

# In[6]:


avoid_set = {
	# 0: [11],
	# 1: [159],
	# 2: [1],
}


# In[7]:


import os
import cluster_model
importlib.reload(cluster_model)

LAYER = 1
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

# In[28]:


importlib.reload(cluster_model)

score_path = [14, 10, 14, 18, 15, 13]
# TODO: embeds to here...
to_score = ds
importlib.reload(cluster_model)
_, top_ds, scores = cluster_model.score_for_neuron(decomp, to_score[:200], LAYER, score_path, top_n=20, weighting_per_layer=[0, 1, 1, 1, 1, 1])


# In[29]:


import numpy as np
# def get_renderable_scores(scored_t: List[str], scores: List


scores_per_token_set = np.array([max(s) for s in scores])
top_args = np.argsort(scores_per_token_set)[::-1]

# scores_per_token_set = [max(s) for s in scores]
# top_args = [s.argmax() for s in scores]

to_score_top = [to_score[i] for i in top_ds]

tokens = [[model.tokenizer.decode(t) for t in model.tokenizer(d)[
    'input_ids']] for d in to_score_top]
tokens_reord = [tokens[i] for i in top_args]
scores_reord = [scores[i] for i in top_args]

# TODO: WHAT IS HAPPENING WITH NAN?
act_simp = [[[[math.exp(tok)]]
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


# ## Try more path like method

# In[22]:


# TODO: SIOMETHING WRONG HERE WITH THE PATHS
import numpy as np
importlib.reload(cluster_model)
score_path = [3, 10, 11, 9, 19, 14]

"""
Paths which seem to have meaning:

"""

weighting_per_layer = np.ones(len(decomp.lattice_scores) + 1)
# TODO: do we need to weight the layers?

# TODO: BY NEURON WEIGHTING!
weighting_per_layer[LAYER] = 2

to_score = [d[:300] for d in ds][:200]
scores = decomp.score(
    to_score,
    score_path=score_path,
    # TODO: how do we weight???
    weighting_per_layer=weighting_per_layer
)

scores_per_token_set = np.array([max(s) for s in scores])
top_args = np.argsort(scores_per_token_set)[::-1]

# TODO: we have the weights here

tokens = [[model.tokenizer.decode(t) for t in model.tokenizer(d)[
    'input_ids']] for d in to_score]
tokens_reord = [tokens[i] for i in top_args]
scores_reord = [scores[i] for i in top_args]


# In[25]:


import math
from IPython.core.display import display, HTML
import math
import numpy as np
from circuitsvis.utils.render import render

# TODO: WHAT IS HAPPENING WITH NAN?
act_simp = [[[[(math.exp(tok * 100))]]
             for tok in s] for s in scores_reord]
act_simp = [[[[((tok * 100))]]
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


# In[26]:


# to_score_simp = ["Solve x + 5 = m * 33 + 4 *n"]
# scores_simp = decomp.score(
#     to_score_simp,
#     score_path=score_path,
#     # TODO: how do we weight???
#     weighting_per_layer=weighting_per_layer
# )

# scores_per_token_set = np.array([max(s) for s in scores_simp])
# top_args = np.argsort(scores_per_token_set)[::-1]

# tokens = [[model.tokenizer.decode(t) for t in model.tokenizer(d)[
#     'input_ids']] for d in to_score]
# tokens_reord = [tokens[i] for i in top_args]
# scores_reord = [scores[i] for i in top_args]
# act_simp = [[[[int(math.exp(tok * 0.001))]]
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


# 

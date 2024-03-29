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


# In[ ]:


from cluster_model import find_max_weight

find_max_weight(decomp.lattice_scores, K=20)


# In[ ]:


import utils
importlib.reload(utils)
utils.top_k_dag_paths_dynamic(decomp.lattice_scores, k=200_000)


# In[ ]:


from circuitsvis.utils.render import RenderedHTML, render

score_path = [8, 57, 89]
score_path = [10, 63, 89]
score_path = [3, 3, 1]
score_path = [5, 4, 6]
"""
Paths which seem to have meaning

[1, 3, 25]: relational words like "if" "in" "soon"
[1, 3, 10]: relational word but now it is more specific to "to"
"""

to_score = [d[:100] for d in ds][:100]
scores = decomp.score(
    to_score,
    score_path=score_path
)


# In[ ]:


import numpy as np

scores_per_token_set = np.array([max(s) for s in scores])
top_args = np.argsort(scores_per_token_set)[::-1]

tokens = [[model.tokenizer.decode(t) for t in model.tokenizer(d)[
    'input_ids']] for d in to_score]
tokens_reord = [tokens[i] for i in top_args]
scores_reord = [scores[i] for i in top_args]
# TODO: SORT!

# TODO sep fun
html = render(
    "TextNeuronActivations",
    tokens=tokens_reord,
    activations=[[[[tok]] for tok in s] for s in scores_reord],
    firstDimensionName="Layer",
    secondDimensionName="Neuron",
    firstDimensionLabels=None,
    secondDimensionLabels=None
)


# In[ ]:


scores_per_token_set, scores_per_token_set[top_args]


# In[ ]:


from IPython.core.display import display, HTML

display(HTML(str(html)))


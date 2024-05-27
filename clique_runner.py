#!/usr/bin/env python
# coding: utf-8

# In[1]:


import src.decorrelate as cluster_model
MODEL_NAME = 'EleutherAI/pythia-70m'
DATASET_NAME = 'NeelNanda/pile-10k'

N_DIMS = 512
SEED = 69_420
 
DEBUG = False
 
if DEBUG:
    N_DATASIZE = 300
    N_BLOCKS = 12
    STRING_SIZE_CUTOFF = 200
else:
    # It gets killed aroun 1_800 idk why. Maybe we have a problem with token truncation somewhere
    N_DATASIZE = 3_000
# 
    # N_CLUSTERS_MIN = int(0.5 * N_DIMS)
    # N_CLUSTERS_MAX = 10 * N_DIMS
    # TODO: DEL ME
    N_BLOCKS = 6
    STRING_SIZE_CUTOFF = 1_200

params = cluster_model.InterpParams(
	lattice_params=cluster_model.LatticeParams(
		top_layer_idx = -1,
        max_n_parents = 4
	),
    # quantization='4bit',
	seed=SEED,
    n_datasize=N_DATASIZE,
    n_blocks=N_BLOCKS,
    model_name=MODEL_NAME,
	model_n_dims=N_DIMS,
    dataset_name=DATASET_NAME,
    string_size_cutoff=STRING_SIZE_CUTOFF,
    quantization='4bit'
)

decomp = cluster_model.Decomposer(params)
decomp.load()


# ## Lets just look at the highest cliques

# In[ ]:


decomp.internal_correlations[1]


# ## Build Up a "Concept" Lattice Using Graph Restrictions
# 
# > TODO: this is for latter...

# In[ ]:


import src.graph as graph
import networkx as nx

LAYER = 3

G = graph.graph_from_correlations(decomp.internal_correlations[LAYER])
MAX_CLIQUE_SIZE = 1_000

GG = graph.sparsify_weighted_graph(G, degree_upperbound=400)
i = nx.find_cliques(GG)
clique = next(i)


# In[ ]:


import importlib
import src.utils as utils
importlib.reload(utils)

# decomp.tune_clique((0.0, clique), 3)


# In[ ]:


# import networkx as nx

# test_at = 100
# GG = graph.sparsify_weighted_graph(G, degree_upperbound=test_at)

# cliques = []

# for node in range(N_DIMS * 2):
#     if node % 50 == 0:
#         print("On node", node)
#     c_it = nx.find_cliques(GG, nodes=[node])
#     c_in_curr = []
#     for c in c_it:
#         c_in_curr.append(c)
#         if len(c_in_curr) > 50:
#             break
#     cliques += c_in_curr
#         # if len(cliques) > 10_000:
#         #     break


# ## Score a cell

# In[ ]:


decomp.internal_correlations[0]


# In[ ]:


import src.decorrelate as decc
import src.kernel as kernel
importlib.reload(decc)
importlib.reload(utils)
importlib.reload(kernel)

LAYER = 2
NEURON = 20

# TODO: we **need** to do something like curr-clique removal so we get more interesting cliques...
# Also the per-neuron thing makes no sense...
decomp.scores_for_neuron(LAYER, NEURON, degree_upperbound=400)


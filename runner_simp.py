#!/usr/bin/env python
# coding: utf-8

# In[2]:


import cluster_model
import numpy as np
from typing import List
import numpy.typing as npt
import kernel

MODEL_NAME = 'EleutherAI/pythia-70m'
DATASET_NAME = 'NeelNanda/pile-10k'
MIXTURE_MODEL_TYPE = "KMenas"
# 
N_DIMS = 512
SEED = 69_420
# 
DEBUG = False
# 
if DEBUG:
    N_DATASIZE = 200
    N_CLUSTERS_MIN = 20
    N_CLUSTERS_MAX = 21
    N_BLOCKS = 6
    STRING_SIZE_CUTOFF = 200
else:
    # It gets killed aroun 1_800 idk why. Maybe we have a problem with token truncation somewhere
    N_DATASIZE = 1_800
# 
    # N_CLUSTERS_MIN = int(0.5 * N_DIMS)
    # N_CLUSTERS_MAX = 10 * N_DIMS
    # TODO: DEL ME
    N_CLUSTERS_MIN = N_DIMS
    N_CLUSTERS_MAX = N_DIMS + 1
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


# In[3]:


decomp.load()


# ## Try a simple pairwise *cancellation* method
# 
# Given the current method of determining signaling, we should get that some stuff sums to one. It would be interesting to plot out what these signals look like.
# We can then use some **approximation** of the neuron via enough signals. Indeed we may consider things on there own as well but not clear
# 
# This eps thing doesn't really work the way we want it to. The problem is that its not like disjoint stuff... maybe tensor programming would actually help here to model the net as a set of correlations
# 
# 
# ## Try Clique Method (or pair method, idk yet) for the specific layer while using individual neurons in following and previous layers to cancel things out

# In[4]:


LAYER = 1


# In[5]:


import numpy as np

eps = 0.05
max_per_layer = 10

pairs_per_layer = []
for i in range(params.n_blocks):
	pairs_per_layer.append([])
	if i == LAYER:
		for idx, corrs in enumerate(decomp.internal_correlations[i]):
			top_n = np.argsort(corrs)[::-1]
			# pairs_per_layer[-1].append([])
			for t in range(max_per_layer):
				x = top_n[t]
				# print(corrs[x])
				if corrs[x] > eps:
					# print("ADDING", corrs[x])
					pairs_per_layer[-1].append((corrs[x], [idx, x]))
	else:
		for idx, _ in enumerate(decomp.internal_correlations[i]):
			pairs_per_layer[-1].append((1, [idx, idx]))


# In[6]:


print(len(pairs_per_layer))
pairs_per_layer[0][:10]


# In[7]:


import importlib
import simplex
import pickle
import os


# In[8]:


import numpy as np
importlib.reload(simplex)

def load_face_corr(layer: int):
	if os.path.exists(f'face_corr_{layer}.npy'):
		return np.load(f'face_corr_{layer}.npy')
	r = simplex.face_correlation_for_layer(layer, pairs_per_layer[layer], pairs_per_layer[layer + 1], decomp.internal_correlations) 
	np.save(f'face_corr_{layer}.npy', r)
	return r

face_corr = [load_face_corr(i) for i in range(params.n_blocks - 1)]

# if os.path.exists('face_corr_0.npy'):
# 	face_corr = [
#        np.load(f'face_corr_{i}.npy')  for i in range(params.n_blocks - 1)
#     ]
# else:
# 		#  TODO: save this
#     face_corr = simplex.face_correlation_lattice(decomp.internal_correlations, pairs_per_layer)
#     for i, f in enumerate(face_corr):
#         np.save(f'face_corr_{i}.npy', f)
# # pickle.dump(clique_lists, open('tmp_for_clique_corrs.pkl', "bw+"))


# In[9]:


(face_corr[1][3] > 0.6).nonzero()[0].shape


# In[29]:


face_corr[0][1]


# In[10]:


import graph
import pickle
import networkx as nx
importlib.reload(graph)

# Hrmmmm.... this type of "disentanglement may just be too harsh?? Or is the harshness a good thing?"
# Hrmmm clearly we need an "all positive thing" going on...
# TODO: there has got to be a way to make this faster...

# TODO: this is too much memory!!! Just recreate using numpy...
if not os.path.exists('tmp_G.pkl'):
    G = graph.GraphOfCorrs(face_corr, corr_cutoff=0.0, max_outgoing_for_feature=8) # Given the current strat we do not need much on the outgoing
    pickle.dump(G, open("tmp_G.pkl", "bw+"))
else:
    G = pickle.load(open('tmp_G.pkl', 'br'))


# In[11]:


list(G.node_idx_to_graph_idx[0].keys())[-10:]


# In[28]:


importlib.reload(graph)

# path_inds = G.get_top_k_paths(0, 1, 10)
self = G
layer = LAYER
neuron = 1
k = 10
path_inds = G.get_top_k_paths(layer, neuron, k, all_disjoint=True,
															from_layer=0, to_layer=3)
path_inds


# In[24]:


paths = [
    [
		pairs_per_layer[layer][idx][1] for layer, idx in enumerate(p[0])
	] for p in path_inds
]
paths


# In[25]:


def score_face_path(embd_dataset: List[npt.NDArray], path: List[simplex.Face],
                    layer: int, from_layer: int, BS=1_024 * 64, cutoff=0.001):

	# TODO: this is the hard part!
    def get_cutoff_for_layer(neuron: int, layer_to_cutoff: int):
        fs = kernel.feature_prob(embd_dataset[layer_to_cutoff], neuron)
        # return (fs.sum() / len(fs)) # TODO: COMP avgs and save somewhere
        nonzeros = fs[np.where(fs > 0)]
        if len(nonzeros) == 0:
            return 0.0
        r = sum(nonzeros) / len(nonzeros) # Need to be above some fraction of the non-zero supported neurons
        return r

    n_tokens = embd_dataset[0].shape[0]
    rets = np.ones(n_tokens)


    for i in range(0, n_tokens, BS):
        top_idx = min(i+BS, n_tokens)
        for p_idx in range(len(path)):
            per_node_score = None
            n_nonzero_on = np.zeros(top_idx - i)
            curr_len = len(path[p_idx])
            curr_layer = p_idx + from_layer
            for node in path[p_idx]:
                # TODO: func out because messy
                local_scores = kernel.feature_prob(embd_dataset[curr_layer][i:top_idx], node)
                c = get_cutoff_for_layer(node, curr_layer)
                # print(c)
                local_scores = local_scores * (local_scores >= c)
                n_nonzero_on += (local_scores > 0)
                
                per_node_score = local_scores if per_node_score is None else per_node_score * local_scores

            rets[i:top_idx] *= (per_node_score * (per_node_score >= cutoff) if layer == curr_layer                 else n_nonzero_on >= curr_len)
    return rets


# In[26]:


paths


# In[27]:


import visualization
Path = List[List[int]]

# TODO: scoring with from_layer
def score_group_vis(paths: List[Path], layer: int, neuron: int, from_layer: int):
    embeds, token_to_original_ds, _ = decomp._get_ds_cache(
        decomp.dataset, decomp.ds_emb)
    BOS_TOKEN = '||BOS||'
    to_vis = []
    for i, path in enumerate(paths):
        print("Scoring path", i, path)
        ret = score_face_path(decomp.ds_emb, path, layer=layer, from_layer=from_layer)
        print(f"ZERO COMPS for feature {i}", ret.nonzero()[0].shape, ret.shape, ret.nonzero()[0].shape[0] / ret.shape[0])
        item_to_scores = {}
        # return log_scores
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

        scores_per_token_set = np.array([max(s) for s in final_scores])
        top_args = np.argsort(scores_per_token_set)[::-1][:100]
        # TODO: BOS?
        tokens = [[BOS_TOKEN] + [decomp.model[1].decode(t) for t in decomp.model[1](d)[
            'input_ids']] for d in decomp.dataset]
        tokens_reord = [tokens[i] for i in top_args]

        scores_reord = [final_scores[i] for i in top_args]
        to_vis.append((tokens_reord, scores_reord))
    visualization.save_display_for_neuron(to_vis, paths=paths, layer=layer, neuron=neuron)

path = paths[1]

k = 8
layer = LAYER
for i, top_ind in enumerate(range(0, 100)):
    from_layer = max(layer - 2, 0)
    to_layer = min(layer + 2, params.n_blocks - 1)
    path_inds = G.get_top_k_paths(layer, top_ind, k, all_disjoint=True, # TODO: hrmm on all disjoint...
                                  from_layer=from_layer, to_layer=to_layer)
    paths = [
        [pairs_per_layer[layer][idx][1] for layer, idx in enumerate(p[0])] for p in path_inds
    ]
    score_group_vis(paths, layer, top_ind, from_layer)
# score_group_vis(path, layer, neuron)


# ## Try looking at internal sub-cliques

# In[ ]:


import networkx as nx
import matplotlib.pyplot as plt
LY=0
f = decomp.internal_correlations[LY].flatten()
f.sort()
plt.plot(f)


# In[ ]:


g_mat = decomp.internal_correlations[LY] * (decomp.internal_correlations[LY] > 0.05)


# In[ ]:


for i in range(len(g_mat)):
	g_mat[i, i] = 0
g = nx.from_numpy_array(g_mat, edge_attr='weight')
nx.draw(g)


# In[ ]:


g.edges(data=True)


# In[ ]:


from typing import List
LIM = 100
cliques_per_node = []
for N in range(1024):
    # print("Getting clique centered at", N)
    cliques_per_node.append([])
    y = nx.find_cliques(g, nodes=[N])
    for i, conns in enumerate(y):
        # print(conns)
        cliques_per_node[-1].append(conns)
        if i >= LIM:
            break


# In[ ]:


import numpy as np

weight_attrs = nx.get_edge_attributes(g, 'weight')
n_tops = 20

def get_avg_clique_weight(clique: List[int]):
    total_weight = 0
    total_cons = 0
    # if len(clique) > 15:
    #     return 0
    # print("GET AVG", len(clique))
    for i in range(len(clique)):
        for j in range(i):
            total_cons += 1
            o = [clique[i], clique[j]]
            o.sort()
            o = tuple(o)
            total_weight += weight_attrs[o]
    return  total_weight / (total_cons ** 2)

top_cliques_per_node = []
for node, cliques in enumerate(cliques_per_node):
    top_cliques_per_node.append([])
    # print("Looking at node", node)
    weights = np.array([get_avg_clique_weight(c) for c in cliques])
    # cliques_per_node[node] = zip(weights, cliques)
    tops = np.argsort(weights)[::-1]
    n = min(len(tops), n_tops)
    for i in range(n):
        # print(tops[i], cliques[tops[i]], weights[tops[i]])
        top_cliques_per_node[-1].append((weights[tops[i]], cliques[tops[i]]))


# In[ ]:


top_cliques_per_node[116]


# ## Keep going

# In[ ]:


from typing import List
import importlib
import utils
importlib.reload(utils)
# modified_lattice = decomp.correlation_scores.copy()

# # TODO: have function which does something like fixing from the a specific neuron?
# # Ie have the sub-lattice with only **related** neurons

# def cutoff_lattice(lattice: List[List[List[float]]], related_cutoff = 1):
#     print(lattice[0].sum())
#     r = [(layer > related_cutoff) * layer for layer in lattice]
#     print(r[0].sum())
#     return r

# def create_sublattice(lattice, layer: int, idx_in_layer: int, related_cutoff = 10):
# 	with_cutoff = cutoff_lattice(lattice, related_cutoff)

# LAST_FEAT = 20
# modified_lattice = cutoff_lattice(decomp.correlation_scores, related_cutoff=15)

# # utils.top_k_dag_paths_dynamic(modified_lattice, k=10_000)


# ## Isolate Specific Neurons

# In[ ]:


LAYER = 0
NEURON = 128


# In[ ]:


import numpy as np
import kernel
from IPython.core.display import display, HTML
from circuitsvis.utils.render import render
import graph
importlib.reload(cluster_model)
importlib.reload(utils)
importlib.reload(graph)
importlib.reload(kernel)

N_CHECK = 4

# decomp.scores_for_neuron(LAYER, NEURON, n_features_per_neuron=3)


# ## Look at distributions of scores

# In[ ]:


import matplotlib.pyplot as plt

# Get distribution of lattice scores

LAYER_START = 4
cutoff = 0.3

# ax = sns.heatmap(decomp.correlation_scores[LAYER_START])
plt.imshow(decomp.correlation_scores[LAYER_START] * (decomp.correlation_scores[LAYER_START] > cutoff), cmap='hot', interpolation='nearest')
plt.show()

plt.plot((decomp.correlation_scores[3] > cutoff).sum(axis=1))


# ## Lets look at the distribution of internal correlations

# In[ ]:


cutoff = 0.2
plt.imshow(decomp.internal_correlations[0] * (decomp.internal_correlations[0] > cutoff), cmap='hot', interpolation='nearest')
plt.show()


# ## Look at the distribution of scores. Is there someway to figure out what the "inclusion cutoff" should be?

# ##

# In[ ]:


import math

feat = (1, 497)
fs = kernel.feature_prob(decomp.ds_emb[feat[0]], feat[1])
fig, ax1 = plt.subplots()
ax2 = ax1.twiny()
fs.sort()

ax1.plot(fs)
log_idx = int(round( math.log(len(fs))))
print(log_idx, len(fs))
nonzeros = fs[np.where(fs > 0)]
ax1.plot(fs[:-log_idx]), sum(fs) / len(fs), sum(nonzeros) / len(nonzeros)


# ## Get scores for layers

# In[ ]:


get_ipython().system('rm cache/correlation-de5f2c593b55f095d11e400fd8f6d0964dc8512c/layer_0_neuron_*')


# In[ ]:


importlib.reload(graph)
decomp.scores_for_layer(0)


# In[ ]:


# TODO: LAYER 0 and NEURON 112... I think that its actually no path on the downstream?
decomp.scores_for_layer(4)


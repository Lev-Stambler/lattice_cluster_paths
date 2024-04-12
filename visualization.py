import os
import json
from typing import List, Tuple
from circuitsvis.utils.render import render

"""
Typescript types we care about:

type BasisDescription = {
  basis: number;
  is_pos: boolean;
};

type LayerFeatures = { [layer: number]: BasisDescription[] }[];

"""


def _get_all_feat_path(layer: int): return f"visualizer/neuron-visualizer/src/lib/db/per_layer/{layer}/features.json"

def add_to_feature_path(layer: int, feature_path: List[List[int]]):
    as_feature_descr = {l: [
        {
            "basis": feat // 2,
            "is_pos": True if feat % 2 == 0 else False
        }
        for feat in feature_path[l]]
        for l in range(len(feature_path))}
    f = _get_all_feat_path(layer)

    if not os.path.exists(f):
        with open(f, "w+") as fp:
            fp.write("[]")

    with open(f, "rw") as fp:
        all_feats: List = json.load(fp)
        all_feats.append(as_feature_descr)
        json.dump(all_feats, fp)

# TODO: use meta tage


def save_path(layer: int, neuron: int):
    return f"built_vis_cite/neurons/layer_{layer}_neuron_{neuron}.htm"


def display_top_scores(tokens_reord, scores_for_path):
    # maxes_per_score = [max(s) for s in scores_for_path]
    # max_score = max(maxes_per_score)
    # print("AAAAAA \n\n", scores_for_path[0][0].shape, scores_for_path[0][0])
    # TODO: probs can reord these better
    act_simp=[[[[tok]]
                 for tok in s] for s in scores_for_path]
    # act_simp = [[[[tok_score]] for tok_score in data_str
    #     ] for data_str in scores_for_paths]
    # act_simp = [[[[tok_score]] for tok_score in s]  for s in scores_for_paths]
    html=render(
        "TextNeuronActivations",
        tokens=tokens_reord,
        activations=act_simp,
        firstDimensionName="Layer",
        secondDimensionName="Neuron",
        firstDimensionLabels=None,
        secondDimensionLabels=None
    )

    return str(html)


def _create_webpage_for_paths_per_neuron(layer: int, neuron: int,
                                         paths_html: List[str], paths: List[List[int]]):
    diff_paths=[]
    for (html, path) in zip(paths_html, paths):
        diff_paths.append(f"""
<link rel="stylesheet" type="text/css" href="/style.css">
<div class='display-wrapper'>
<h3> Path: {path} </h3>
{html}
</div>
</div>
<br />
<hr />
""")
    joined='\n'.join(diff_paths)
    html=f"""
    	<div class="page-wrapper">
        <h2>Analysis for Layer {layer} and Neuron {neuron}</h2>
        {joined}
        </div>
    """
    with open(save_path(layer, neuron), "w") as f:
        f.write(html)


def save_display_for_neuron(scores_for_paths: List[Tuple[List, List]], paths: List[List[int]], layer: int, neuron: int):
    htmls=[display_top_scores(s[0], s[1]) for s in scores_for_paths]
    _create_webpage_for_paths_per_neuron(layer, neuron, htmls, paths)

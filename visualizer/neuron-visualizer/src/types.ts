type BasisDescription = {
  basis: number;
  is_pos: boolean;
};

type LayerFeatures = { [layer: number]: BasisDescription[] }[];

export interface Metadata {
	n_layers: number,
  // layer_to_features: LayerFeatures[];
}

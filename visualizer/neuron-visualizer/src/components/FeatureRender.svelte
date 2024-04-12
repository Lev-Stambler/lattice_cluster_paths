<script lang="ts">
  import type { FeatureSet } from "../types";
  import _all_features from "$lib/db/all_features.json";
  import SvelteTable from "svelte-table";

  export let layer: number;

  const all_features: FeatureSet = _all_features;

  const rows = all_features.layer_to_features[layer];

  const keys = Array(all_features.n_layers)
    .fill(0)
    .map((_, i) => ({
      key: i,
      value: (v: any) => v[i],
      title: `Clique for layer ${i + 1}`,
      sortable: false,
      // filterOptions: (rows: any) => {
      //   // TODO: FILTER?
      //   return rows;
      // },
				//  TODO: hrmm
      // filterValue: (v: any) => v.first_name.charAt(1).toLowerCase(),
    }));

  const columns = keys;
</script>

<SvelteTable {columns} {rows}></SvelteTable>

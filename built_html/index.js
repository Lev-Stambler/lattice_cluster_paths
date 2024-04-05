const load_neurons = async () => {
  const n_layers = 6;
  const n_neurons = 512;
  const flattened = Array(n_layers * n_neurons)
    .fill(0)
    .map((_, i) => [
      Math.floor(i / n_neurons), // layer
      i % n_neurons, // neuron
    ]);

  const proms = flattened.map(async ([layer, neuron]) => {
    try {
      const r = await fetch(
        `neurons/layer_${layer}_neuron_${neuron}.htm`
      );
      if (r.ok) {
				return `<li><a href="neurons/layer_${layer}_neuron_${neuron}.htm">Neuron ${neuron} in layer ${layer}</a></li>`
      }
				 return ""
      return `Neuron ${neuron} in layer ${layer} not found`;
    } catch (error) {
				 return ""
      return `Neuron ${neuron} in layer ${layer} not found`;
    }
  });
  const all_inner = await Promise.all(proms);
  document.getElementById("neuron_inspect").innerHTML = all_inner.join("\n");
};

load_neurons()
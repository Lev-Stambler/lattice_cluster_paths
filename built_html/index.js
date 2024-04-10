const load_neurons = async () => {
  window.onload = async () => {
    const n_layers = 6;
    const n_neurons = 1024;
    const flattened = Array(n_layers * n_neurons)
      .fill(0)
      .map((_, i) => [
        Math.floor(i / n_neurons), // layer
        i % n_neurons, // neuron
      ]);

    const per_layer = (layer) => {
      return `
      <div class="layer-wrapper">
      <h2>Layer ${layer}</h2>
      <details>
        <summary>Neurons for layer ${layer}</summary>
        <br />
        <div class="layer-bullets">
          ${Array(n_neurons)
            .fill(0)
            .map(
              (_, i) =>
                `<button onclick="(() => {window.location.href='neurons/layer_${layer}_neuron_${i}.htm'})()">Neuron ${i}</button>`
            )
            .join("\n")}
        </div>
      </details>
      </div>
      `;
    };

    const layers = Array(n_layers)
      .fill(0)
      .map((_, i) => per_layer(i));

    document.getElementById("neuron_inspect").innerHTML = layers.join("\n");
  };
};

load_neurons();

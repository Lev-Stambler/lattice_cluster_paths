<script lang="ts">
  export let activations: number[][];
  export let tokens: string[][]; // TODO: maybe unicode instead

  console.assert(tokens.length === activations.length);
  const showNumbers = activations.map((a) => a.map((_) => false));
</script>

{#each activations as acts, i}
  <div class="string_wrapper">
    {#each acts as act, j}
      <div
        class="tok_render"
        style={`background-color: rgb(255, 255, 255 - ${Math.floor(act * 255)}`}
        on:mouseover={() => (showNumber = true)}
        on:mouseout={() => (showNumber = false)}
      >
        {#if showNumbers[i][j]}
          <span class="number-display {showNumbers[i][j] ? 'show' : ''}">
            {act}
          </span>
        {:else}
          {tokens[i][j]}
        {/if}
      </div>
    {/each}
  </div>
  <!-- content here -->
{/each}

<style>
  .tok_render {
    padding: 3px;
  }
</style>

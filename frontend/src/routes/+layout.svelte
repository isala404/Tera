<script lang="ts">
  import { ForgeProvider } from "@forge/svelte";
  import { PUBLIC_API_URL } from "$env/static/public";

  interface Props {
    children: import("svelte").Snippet;
  }

  let { children }: Props = $props();
</script>

<ForgeProvider url={PUBLIC_API_URL}>
  {@render children()}
</ForgeProvider>

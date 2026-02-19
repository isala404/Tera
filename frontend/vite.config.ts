import { sveltekit } from "@sveltejs/kit/vite";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [sveltekit()],
  server: {
    // Enable polling for Docker/container environments
    watch: {
      usePolling: true,
    },
  },
});

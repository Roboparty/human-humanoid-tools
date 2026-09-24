import { fileURLToPath, URL } from "node:url";

import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "./",
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    host: "127.0.0.1",
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8009",
    },
  },
  build: {
    // FastAPI and Electron both load this one renderer build.
    outDir: "../static",
    emptyOutDir: true,
    // Keep large framework/rendering vendors cache-stable. Core workspaces ship
    // eagerly; the optional Video-to-Motion surfaces remain on-demand.
    chunkSizeWarningLimit: 1_000,
    rolldownOptions: {
      output: {
        codeSplitting: {
          groups: [
            {
              name: "vendor-react",
              test: /node_modules[\\/](?:react|react-dom|scheduler)[\\/]/,
              priority: 30,
            },
            {
              name: "vendor-three",
              test: /node_modules[\\/](?:three|@react-three)[\\/]/,
              priority: 20,
            },
            {
              name: "vendor-ui",
              test:
                /node_modules[\\/](?:@radix-ui|clsx|tailwind-merge|class-variance-authority)[\\/]/,
              priority: 10,
            },
          ],
        },
      },
    },
  },
});

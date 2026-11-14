import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// In development the UI runs here and proxies the API to the control plane; a build is served by the control plane.
const api = process.env.CONTROLPLANE_URL ?? `http://127.0.0.1:${process.env.CONTROLPLANE_PORT ?? 8000}`;

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": new URL("./src", import.meta.url).pathname } },
  build: { chunkSizeWarningLimit: 1024 },
  server: { host: "127.0.0.1", port: Number(process.env.UI_PORT ?? 5173), strictPort: true, proxy: { "/api": api } },
});

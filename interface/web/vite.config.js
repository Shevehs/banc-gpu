import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// The Python server (interface/server.py) serves dist/ and the /api routes; in development, Vite proxies /api to it.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "../dist", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});

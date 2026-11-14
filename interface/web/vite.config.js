import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// Le serveur Python (interface/serveur.py) sert dist/ et l'API /api ; en développement, Vite relaie /api vers lui.
export default defineConfig({
  plugins: [react()],
  base: "./",
  build: { outDir: "../dist", emptyOutDir: true },
  server: { proxy: { "/api": "http://127.0.0.1:8765" } },
});

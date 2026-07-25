import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/jobs": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/output": "http://localhost:8000",
    },
  },
});

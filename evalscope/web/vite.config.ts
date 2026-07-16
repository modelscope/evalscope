import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    // Emit `.vite/manifest.json` so the size-budget script (scripts/sizeBudget.ts)
    // can statically identify the initial-load chunk graph (entry + its static
    // imports, excluding lazy/dynamic-import chunks) and enforce the gzip budget
    // (Req 16.6). Additive: does not change bundling behaviour.
    manifest: true,
  },
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  server: {
    port: 5173,
    host: '0.0.0.0',
    proxy: {
      '/api/v1': {
        target: 'http://127.0.0.1:9000',
        changeOrigin: true,
      },
      '/health': {
        target: 'http://127.0.0.1:9000',
        changeOrigin: true,
      },
    },
  },
})

import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

export default defineConfig({
  plugins: [react(), tailwindcss()],
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

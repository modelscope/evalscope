import { defineConfig, type Plugin } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

const KATEX_FONT_FALLBACKS =
  /,url\([^)]*\.woff\) format\(["']woff["']\),url\([^)]*\.ttf\) format\(["']truetype["']\)/g

function katexWoff2Only(): Plugin {
  return {
    name: 'katex-woff2-only',
    enforce: 'pre',
    transform(code, id) {
      if (!id.includes('/katex/dist/katex.min.css')) return null
      const woff2Only = code.replace(KATEX_FONT_FALLBACKS, '')
      if (woff2Only === code) this.error('KaTeX font fallbacks were not found; update the WOFF2-only transform.')
      return woff2Only
    },
  }
}

export default defineConfig({
  plugins: [katexWoff2Only(), react(), tailwindcss()],
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

import path from 'node:path';
import { fileURLToPath } from 'node:url';

import sitemap from '@astrojs/sitemap';
import tailwindcss from '@tailwindcss/vite';
import { defineConfig, fontProviders } from 'astro/config';
import icon from 'astro-icon';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export default defineConfig({
  output: 'static',
  site: 'https://modelscope.github.io',
  base: '/evalscope',
  fonts: [
    {
      provider: fontProviders.fontsource(),
      name: 'Inter',
      cssVariable: '--font-inter',
      weights: ['400 900'],
      styles: ['normal'],
      subsets: ['latin'],
      fallbacks: ['sans-serif'],
    },
  ],
  integrations: [sitemap(), icon({ include: { tabler: ['*'] } })],
  vite: {
    plugins: [tailwindcss()],
    resolve: { alias: { '~': path.resolve(__dirname, './src') } },
  },
});

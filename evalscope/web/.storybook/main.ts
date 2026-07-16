import { dirname, resolve } from 'path'
import { fileURLToPath } from 'url'
import { mergeConfig } from 'vite'
import type { StorybookConfig } from '@storybook/react-vite'

const __dirname = dirname(fileURLToPath(import.meta.url))

/**
 * Storybook configuration for the EvalScope Web Console.
 *
 * Uses the '@storybook/react-vite' framework so the project's own
 * vite.config.ts (React + Tailwind plugins) is reused automatically. The
 * viteFinal hook only re-declares the '@' path alias to keep story imports
 * consistent with the app.
 */
const config: StorybookConfig = {
  stories: ['../src/**/*.stories.@(ts|tsx)'],
  framework: {
    name: '@storybook/react-vite',
    options: {},
  },
  core: {
    disableTelemetry: true,
  },
  async viteFinal(viteConfig) {
    return mergeConfig(viteConfig, {
      resolve: {
        alias: {
          '@': resolve(__dirname, '../src'),
        },
      },
    })
  },
}

export default config

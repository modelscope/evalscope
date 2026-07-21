import { defineConfig, mergeConfig } from 'vitest/config'
import viteConfig from './vite.config'

// Vitest configuration for the Web Console.
// Reuses the existing Vite config (including the `@` path alias) so that tests
// resolve modules exactly like the application build does.
export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      // Enable a browser-like DOM so React Testing Library can render components.
      environment: 'jsdom',
      // Expose `describe`/`it`/`expect`/`vi` without explicit imports.
      globals: true,
      // Deterministic setup: fake timers, fixed system time and a fixed
      // fast-check seed, plus network access disabled.
      setupFiles: ['./src/test/setup.ts'],
      // Only treat co-located test/spec files as tests.
      include: ['src/**/*.{test,spec}.{ts,tsx}'],
      coverage: {
        provider: 'v8',
        reporter: ['text', 'html', 'lcov'],
        include: ['src/**/*.{ts,tsx}'],
        exclude: [
          'src/**/*.{test,spec}.{ts,tsx}',
          'src/test/**',
        ],
      },
    },
  }),
)

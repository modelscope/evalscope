import js from '@eslint/js'
import globals from 'globals'
import reactHooks from 'eslint-plugin-react-hooks'
import reactRefresh from 'eslint-plugin-react-refresh'
import tseslint from 'typescript-eslint'
import { defineConfig, globalIgnores } from 'eslint/config'

export default defineConfig([
  // `coverage/` holds the v8 reporter's own bundled scripts, which are neither
  // authored here nor lintable against this config.
  globalIgnores(['dist', 'test-results', 'coverage']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      js.configs.recommended,
      tseslint.configs.recommended,
      reactHooks.configs.flat.recommended,
      reactRefresh.configs.vite,
    ],
    languageOptions: {
      globals: globals.browser,
    },
  },
  {
    files: ['src/components/**/*.{ts,tsx}', 'src/pages/**/*.{ts,tsx}'],
    rules: {
      'no-restricted-syntax': [
        'error',
        {
          selector: 'CallExpression[callee.property.name="toFixed"]',
          message: 'Format metrics through the centralized metric registry instead of calling toFixed directly.',
        },
      ],
    },
  },
  {
    files: [
      'src/domain/**/*.ts',
    ],
    rules: {
      'no-restricted-imports': [
        'error',
        {
          patterns: [
            {
              group: [
                'react',
                'react/*',
                'react-dom',
                'react-dom/*',
                'react-router',
                'react-router-dom',
                '@/components/*',
                '@/pages/*',
                '**/components/*',
                '**/pages/*',
              ],
              message: 'Domain data models must not depend on the rendering layer.',
            },
          ],
        },
      ],
    },
  },
])

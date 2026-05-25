# EvalScope Web Dashboard

React + TypeScript + Vite frontend for EvalScope, served by `evalscope service`.

## Design system

All visual decisions live in [DESIGN.md](../../DESIGN.md) at the repo root. Code MUST express styling
through the token + atom layer there — never through hex literals, inline hex-tinted shadows, or
hand-written eyebrow class strings.

### Signature atoms (`src/components/ui/`)

| Import | Purpose | DESIGN.md ref |
|---|---|---|
| `import ScoreChip from '@/components/ui/ScoreChip'` | Small caption-mono score pill | `{components.score-chip}` |
| `import ScoreBadge from '@/components/ui/ScoreBadge'` | Bold percentage / boolean pill | `{components.score-badge}` |
| `import KpiCard from '@/components/ui/KpiCard'` | Dashboard hero metric tile | `{components.kpi-card}` |
| `import Eyebrow from '@/components/ui/Eyebrow'` | UPPERCASE section eyebrow label | `{typography.label-xs}` |
| `import ChatBubble from '@/components/ui/ChatBubble'` | 5-role chat surface (user / bot / tool / reasoning / system) | `{components.chat-bubble}` |
| `import EmptyState from '@/components/common/EmptyState'` | Empty / welcome state — 64×64 icon tile + 2 lines | `{components.empty-state}` |
| `import PathBar from '@/components/ui/PathBar'` | Dashboard "scan this directory" input row | `{components.path-bar}` |
| `import EvalRunCard from '@/components/ui/EvalRunCard'` | Eval-row in the dashboard timeline | `{components.eval-run-card}` |
| `import ModelGroupHeader from '@/components/ui/ModelGroupHeader'` | Collapsible model / dataset group header | `{ex-model-group-header}` |

### Typography utility classes (defined in `src/index.css`)

Apply via Tailwind class — these cover every DESIGN.md `{typography.*}` token. Never re-derive
`text-xs font-semibold uppercase tracking-wider …` by hand:

| Class | Token |
|---|---|
| `.type-display-xl` | `{typography.display-xl}` — 24px / bold / tracking-tight |
| `.type-title-md` | `{typography.title-md}` — 16px / bold |
| `.type-body-sm` | `{typography.body-sm}` — 14px / regular |
| `.type-body-sm-strong` | `{typography.body-sm-strong}` — 14px / medium |
| `.type-body-xs` | `{typography.body-xs}` — 12px / regular |
| `.type-label-xs` | `{typography.label-xs}` — 12px / semibold / UPPERCASE / tracking-wider / text-muted |
| `.type-table-xs` | `{typography.table-xs}` — 10px / semibold / UPPERCASE / tracking-wider / text-dim |
| `.type-caption-mono` | `{typography.caption-mono}` — 12px / mono / tabular-nums |
| `.type-code` | `{typography.code}` — 13px / mono |
| `.type-button-sm` | `{typography.button-sm}` — 12px / medium |
| `.type-button-md` | `{typography.button-md}` — 14px / medium |
| `.type-button-lg` | `{typography.button-lg}` — 16px / medium |

### Score color helpers

`scoreColor(s)` / `scoreBg(s)` from `@/utils/colorScale` are the single source for the HSL score
gradient (`hsl(score × 120, 70%, 45%)`).

### Rules

- No inline `style={{ background: 'var(--xxx)' }}` when a Tailwind class would do.
- No hand-written hex literals or `rgba(129,109,248,…)` glows — go through `--shadow-glow*`.
- `text-[var(--text-dim)]` requires an inline `// text-dim allowed` comment per DESIGN.md §Text.
- Compare view tops out at 3 model slots (`MAX_COMPARE_SLOTS` in `ComparePage.tsx`) — adding a 4th
  brand color violates DESIGN.md §Compare Slots.

---

## Vite scaffold notes

This project started from the Vite React + TypeScript template:

Currently, two official Vite React plugins are available:

- [@vitejs/plugin-react](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react) uses [Oxc](https://oxc.rs)
- [@vitejs/plugin-react-swc](https://github.com/vitejs/vite-plugin-react/blob/main/packages/plugin-react-swc) uses [SWC](https://swc.rs/)

## React Compiler

The React Compiler is not enabled on this template because of its impact on dev & build performances. To add it, see [this documentation](https://react.dev/learn/react-compiler/installation).

## Expanding the ESLint configuration

If you are developing a production application, we recommend updating the configuration to enable type-aware lint rules:

```js
export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...

      // Remove tseslint.configs.recommended and replace with this
      tseslint.configs.recommendedTypeChecked,
      // Alternatively, use this for stricter rules
      tseslint.configs.strictTypeChecked,
      // Optionally, add this for stylistic rules
      tseslint.configs.stylisticTypeChecked,

      // Other configs...
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

You can also install [eslint-plugin-react-x](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-x) and [eslint-plugin-react-dom](https://github.com/Rel1cx/eslint-react/tree/main/packages/plugins/eslint-plugin-react-dom) for React-specific lint rules:

```js
// eslint.config.js
import reactX from 'eslint-plugin-react-x'
import reactDom from 'eslint-plugin-react-dom'

export default defineConfig([
  globalIgnores(['dist']),
  {
    files: ['**/*.{ts,tsx}'],
    extends: [
      // Other configs...
      // Enable lint rules for React
      reactX.configs['recommended-typescript'],
      // Enable lint rules for React DOM
      reactDom.configs.recommended,
    ],
    languageOptions: {
      parserOptions: {
        project: ['./tsconfig.node.json', './tsconfig.app.json'],
        tsconfigRootDir: import.meta.dirname,
      },
      // other options...
    },
  },
])
```

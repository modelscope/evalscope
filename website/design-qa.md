# Design QA

## Scope

- Local preview: `http://127.0.0.1:4321/evalscope/`
- Browser: Codex in-app browser
- Design system: warm paper ground, black display type, technical blue labels, gold measurement marks, fine rulers and registration lines.
- Evidence: only local EvalScope reports and documentation captures are used. No generated product screenshots, invented metrics, local-model downloads or GPU jobs were used.

## Current structure

- Home is the long-form product path: Overview → Evaluate → Agent & Harness → Performance → Benchmarks → Visualization → Get Started. Research remains a separate exploratory route.
- The fixed Home directory follows the visible chapter, exposes the current position and links to every Home section. Each Home chapter includes a direct route to its full page.
- The header Home menu opens the full product routes; Benchmarks, Research and Use with AI always use their standalone routes.
- The Home hero pairs one runnable API-first evaluation command with a real GSM8K result frame. Its quiet measurement band is retained, while the redundant four-cell summary strip is removed.
- The three product-surface cards use a single combined sequence-and-icon mark instead of separate number and icon ornaments.
- Get Started begins with a copyable, English Skill-installation prompt for an AI coding agent, then offers task-specific prompts and an annotated command-line quickstart. The same quickstart panel is available on the standalone guide.
- Home Agent and Performance chapters reserve a wider evidence column on desktop so their report captures remain readable at a glance.
- Benchmark rows use native disclosure: users can expand an official Overview in place and choose the documentation link only when they need the full record.

## Copy and localization

- English and Chinese communicate the same bounded, API-first workflow: use existing endpoint configuration, run five GSM8K samples, preserve artifacts and report missing prerequisites plainly. Every prompt copied to an AI stays in English for reliable agent execution.
- Chinese copy uses `API 端点`, `提示词`, `AI 编程助手` and `元数据` consistently; environment variable names, commands and product APIs remain unchanged.
- Copy controls are localized and restore their original label after the success state in both locales.

## Interaction and accessibility

- The Home directory updates its active chapter while scrolling and retains anchor navigation.
- Evidence stages remain focusable, expose their active state and update the live inspector.
- Motion is low-amplitude: active evidence cards lift slightly and respond to pointer position. All reveal, stage and pointer effects are disabled under `prefers-reduced-motion`.
- Every page hero restores the same low-contrast blue pointer glow. It follows the cursor beneath content and remains disabled under `prefers-reduced-motion`.
- Shared `CodeSnippet`, `PromptCard` and `QuickstartPanel` components provide consistent copy controls and command treatment.

## Browser review

- Home: confirmed the removed instrument strip is absent; the measurement ruler, three surface cards, chapter directory, real evidence frames and both Get Started paths remain available.
- Standalone Get Started: confirmed the four task prompts, localized copy actions and annotated terminal quickstart render together.
- Existing route QA covers Evaluation, Agent & Harness, Performance, Benchmarks, Visualization and Research: each uses its intended real evidence and the restrained reference decoration without text collisions.

## Verification

- `npm run fix`
- `npm run verify` — Astro diagnostics, ESLint, Prettier, 256 benchmark metadata records and static build of 17 routes including 404.
- `git diff --check`

Current result: reviewed and ready to commit after the final verification refresh.

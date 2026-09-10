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

---

## Architecture diagram native implementation QA

**Comparison target**

- Source visual truth: `website/src/assets/illustrations/architecture-paper-drafts/architecture-paper-draft-final-backends-bottom-right.png` (1672 × 941 px).
- Implementation: `http://127.0.0.1:4321/evalscope/#evaluation`, rendered in the Codex in-app browser at a 1250 × 1050 px desktop capture. The component keeps a 970 px minimum board width and scrolls horizontally on narrower screens; no density normalization was required for this desktop review.
- State: Home → Evaluate chapter, default light theme. The latest browser capture shows the complete three-column board, both lower workflows, their rightward output arrows, and the optional-backends note.
- Evidence: source image was opened locally; the browser-rendered implementation was captured in the same review session. Focused comparison covered the board title, three primary columns, lower workflow rows, and lower-right annotation. A persistent Codex overlay obscures a small lower center area of the desktop capture, but does not cover the inspected labels or the diagram bounds.

**Findings**

- No actionable P0/P1/P2 differences remain. The implementation intentionally translates the selected raster design into semantic HTML cards and Tabler icons so its labels remain selectable, localizable, and accessible rather than trying to reproduce raster art pixel-for-pixel.
- [P3] The native core-flow labels are necessarily more compact than the source illustration, but the post-fix desktop capture keeps every card inside the core panel and avoids clipped content or an unnecessary horizontal scrollbar.

**Required fidelity surfaces**

- Fonts and typography: the existing EvalScope display and mono treatments are retained; the board title, panel labels and small workflow copy remain legible without truncation in the inspected desktop state.
- Spacing and layout rhythm: the paper frame, three primary columns, two lower workflow rows and right-aligned optional-backend note are preserved. The #3 artifacts panel spans all three diagram rows; this tightens the core region and gives both lower workflows a direct visual destination. The redundant board title is removed in favor of a compact lower-left caption.
- Colors and visual tokens: paper, ink, blue, green and gold derive from the site tokens; blue marks native evaluation, green marks inputs, and the quiet neutral artifacts panel avoids over-emphasizing optional backends.
- Image quality and asset fidelity: no raster UI or handmade SVG was substituted. The diagram uses the existing Tabler icon library for semantic symbols; the selected source illustration remains retained as visual provenance, not as the page UI.
- Copy and content: the diagram uses `Model`, not “Lazy Model”; it includes the previously missing Agent & Harness and Serving Performance paths, and places OpenCompass, VLMEvalKit and RAG Eval only as a lower-right optional-backends annotation.

**Implementation checklist**

- [x] Replace the Home Evaluate inspector with a native semantic architecture diagram.
- [x] Keep the full native evaluation path and inspectable artifact destinations visible.
- [x] Add Agent & Harness and Serving Performance flows before their report destinations.
- [x] Keep external backends visually secondary in the lower-right corner.
- [x] Render and inspect the Home Evaluate state in a real browser.

**Comparison history**

- Initial native render: browser comments identified [P1] a compressed core flow with excess vertical whitespace, and [P1] Agent/Serving arrows that did not visibly terminate at the artifacts destination.
- Fix: widened the central column while retaining a 970 px board floor, reduced the top-row panel height, made #3 span all diagram rows, and replaced diagonal endpoint marks with horizontal arrows that extend into the #3 boundary. Optional backends moved into the lower portion of #3.
- Post-fix evidence: the latest Codex in-app-browser desktop capture shows the entire core flow within its panel and both lower-row arrows pointing directly into the spanning Inspectable Artifacts panel.
- Follow-up fix: removed the repeated board heading after browser review and placed `Fig. 01 · EvalScope Architecture` as the lower-left caption. The final browser capture confirms the diagram now begins directly with the three-column architecture.
- Layout refinement: browser review identified [P2] disproportionate lower whitespace in the native-evaluation panel, a mismatched left baseline between the overview and Evaluate headings, and excessive space before the Agent chapter. Removed the fixed panel height, centered the core flow inside its available grid track, compacted the input cards, aligned the Evaluate heading with the overview baseline on desktop, and reduced the Evaluate bottom padding from 92 px to 48 px. The browser review confirms the shorter board and tighter chapter transition without clipped flow content.
- Content-rail correction: a full Home audit found two competing desktop container rules: generic sections used a 1180 px centered rail while product chapters used a 1060 px rail with independent exceptions. Home sections now share one 1060 px desktop content rail, left-aligned at `max(20px, (viewport - 1180px) / 2)` to reserve the right-side chapter directory. The local Evaluate heading offset was removed, so Overview, Evaluate, Agent, Performance, Benchmarks, Visualization and Get Started use the same title baseline.

final result: passed

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
- Implementation: `http://127.0.0.1:4321/evalscope/#evaluation`, rendered in the Codex in-app browser at a 1200 × 1066 px desktop capture. The component keeps a 1000 px minimum board width and scrolls horizontally on narrower screens; no density normalization was required for this desktop review.
- State: Home → Evaluate chapter, default light theme. The latest browser capture shows the complete three-column board, both lower workflows, their rightward output arrows, the core-panel optional-backends note, and the rebalanced artifact tiles.
- Evidence: source image was opened locally; the browser-rendered implementation was captured in the same review session. Focused comparison covered the board title, three primary columns, lower workflow rows, the lower-left core annotation, and the artifact-card grid. A persistent Codex overlay obscures a small lower center area of the desktop capture, but does not cover the inspected labels or the diagram bounds.

**Findings**

- No actionable P0/P1/P2 differences remain. The implementation intentionally translates the selected raster design into semantic HTML cards and Tabler icons so its labels remain selectable, localizable, and accessible rather than trying to reproduce raster art pixel-for-pixel.
- [P3] The native core-flow labels are necessarily more compact than the source illustration, but the post-fix desktop capture keeps every card inside the core panel and avoids clipped content or an unnecessary horizontal scrollbar.

**Required fidelity surfaces**

- Fonts and typography: the existing EvalScope display and mono treatments are retained; the board title, panel labels and small workflow copy remain legible without truncation in the inspected desktop state.
- Spacing and layout rhythm: the paper frame, three primary columns and two lower workflow rows are preserved. The #3 artifacts panel spans all three diagram rows and now uses a balanced two-by-three tile grid; the optional-backends note sits quietly at the lower left of #2. The redundant board title is removed in favor of a compact lower-left caption.
- Colors and visual tokens: paper, ink, blue, green and gold derive from the site tokens; blue marks native evaluation, green marks inputs, and the quiet neutral artifacts panel avoids over-emphasizing optional backends.
- Image quality and asset fidelity: no raster UI or handmade SVG was substituted. The diagram uses the existing Tabler icon library for semantic symbols; the selected source illustration remains retained as visual provenance, not as the page UI.
- Copy and content: the diagram uses `Model`, not “Lazy Model”; it includes the previously missing Agent & Harness and Serving Performance paths, and places OpenCompass, VLMEvalKit and RAG Eval only as a lower-left optional-backends annotation within native evaluation.

**Implementation checklist**

- [x] Replace the Home Evaluate inspector with a native semantic architecture diagram.
- [x] Keep the full native evaluation path and inspectable artifact destinations visible.
- [x] Add Agent & Harness and Serving Performance flows before their report destinations.
- [x] Keep external backends visually secondary in the lower-left corner of native evaluation.
- [x] Render and inspect the Home Evaluate state in a real browser.

**Comparison history**

- Initial native render: browser comments identified [P1] a compressed core flow with excess vertical whitespace, and [P1] Agent/Serving arrows that did not visibly terminate at the artifacts destination.
- Fix: widened the central column while retaining a 970 px board floor, reduced the top-row panel height, made #3 span all diagram rows, and replaced diagonal endpoint marks with horizontal arrows that extend into the #3 boundary. Optional backends moved into the lower portion of #3.
- Post-fix evidence: the latest Codex in-app-browser desktop capture shows the entire core flow within its panel and both lower-row arrows pointing directly into the spanning Inspectable Artifacts panel.
- Follow-up fix: removed the repeated board heading after browser review and placed `Fig. 01 · EvalScope Architecture` as the lower-left caption. The final browser capture confirms the diagram now begins directly with the three-column architecture.
- Layout refinement: browser review identified [P2] disproportionate lower whitespace in the native-evaluation panel, a mismatched left baseline between the overview and Evaluate headings, and excessive space before the Agent chapter. Removed the fixed panel height, centered the core flow inside its available grid track, compacted the input cards, aligned the Evaluate heading with the overview baseline on desktop, and reduced the Evaluate bottom padding from 92 px to 48 px. The browser review confirms the shorter board and tighter chapter transition without clipped flow content.
- Content-rail correction: a full Home audit found two competing desktop container rules: generic sections used a 1180 px centered rail while product chapters used a 1060 px rail with independent exceptions. Home sections now share one 1060 px desktop content rail, left-aligned at `max(20px, (viewport - 1180px) / 2)` to reserve the right-side chapter directory. The local Evaluate heading offset was removed, so Overview, Evaluate, Agent, Performance, Benchmarks, Visualization and Get Started use the same title baseline.
- Follow-up annotation pass: browser comments identified [P2] flattened number badges and a crowded #3 artifact region. Made panel-number badges fixed circular flex items; moved optional backends to the lower-left of #2; and changed #3 to a two-by-three grid of centered artifact tiles. The final 1200 × 1066 in-app-browser capture confirms circular 1/2/3 badges, readable artifact labels, and clear separation between core options and inspectable outputs.
- Workflow spacing pass: browser comments identified [P2] inconsistent density in the Agent & Harness and Serving Performance rows. Increased the board floor enough to protect the title column, restored the single-line workflow headings, and rebalanced each row into a fixed heading rail, four evenly sized steps, consistently spaced arrows, and one output arrow. The final 1200 × 1066 in-app-browser capture confirms no mid-word breaks in `Tools & Environment` or `TTFT · RPS · Throughput`, while both rows retain a clear path into #3.

final result: passed

---

## Recorded CLI replay QA

**Evidence provenance**

- Every displayed value comes from a fresh local run rooted at `outputs/website-terminal-demo/`; that root and its raw logs are ignored by Git. The website only commits the redacted replay rows in `src/data/terminal-demo.ts`.
- `eval` run ID `20260911_103219`: `qwen-plus`, native `openai_api`, GSM8K `main` plus ARC `ARC-Easy` / `ARC-Challenge`, `--limit 5 --seed 42 --generation-config stream=True`. Source files: `eval-multi-stream-cli.log`, `20260911_103219/reports/report.html`, and the matching `predictions/`, `reviews/`, and `configs/` directories. Displayed: 15 samples; GSM8K Accuracy 100% (average latency 3.768 s, TTFT 587.2 ms, throughput 49.62 tok/s); ARC Accuracy 100% / 80% / 90% overall (average latency 0.496 s, TTFT 400.8 ms, throughput 8.06 tok/s). The frozen task config confirms `stream: true`.
- `perf` run ID `20260911_103241`: `qwen-plus`, OpenAI-compatible DashScope API, random 1,024-token workload, `--stream`, 3 warmup requests and 24 measured requests for each parallelism tier. Source files: `perf-stream-cli.log`, `20260911_103241/qwen-plus/performance_summary.txt`, and `perf_report.html`. Displayed: parallel 1 / 2 / 4 respectively report RPS 0.47 / 0.79 / 1.42, average TTFT 433.25 / 446.91 / 405.09 ms, decode throughput 58.45 / 50.23 / 50.74 tok/s, workload throughput 47 / 79.41 / 141.66 tok/s, and 100% success for every tier. The run generated 7,200 tokens at an average output rate of 73.3 tok/s.
- `service` run ID `20260911_101202`: `evalscope service --host 127.0.0.1 --port 9000 --outputs outputs/website-terminal-demo`. Source file: `service-cli.log`; the captured service printed `http://127.0.0.1:9000/dashboard`, and `GET /health` returned `{"service":"evalscope","status":"ok"}` at 10:12:13 +08:00 before the locally started process was stopped.
- Commands in the site retain the exact CLI flags and output roots used for successful runs. The secret value, Authorization header, and local absolute tokenizer path are redacted; no prompt text or model completion is displayed.

**Interaction and layout review**

- The replay is placed inside `#evaluation`, directly after the architecture figure and before the Agent chapter. Its dark terminal, gold measurement edge, toolbar dots, compact mono labels, and copy treatment reuse the existing Home visual language without changing `CodeSnippet`.
- Desktop browser review: `http://127.0.0.1:4321/evalscope/?terminal-replay=real-logs#recorded-runs` in the Codex in-app browser at 1200 px. The architecture ends before the `A CLI demo, with real output.` heading; the terminal fits the content rail, leaves the right-side section index visible, and the Agent chapter starts cleanly below it.
- Browser replay review: the default `eval` screen renders one continuous terminal transcript: the copied command occurs once, then real timestamped streaming CLI lines, a single overwritten `Running[eval]` line with source 0% / 50% / 100% snapshots, GSM8K and ARC report tables, both original perf-table headers and separators (including Avg Lat / TTFT / TPOT / throughput / input / output columns), and the saved report path. Every ASCII/Unicode table is one `pre` block rather than separate row elements, matching native-terminal line continuity. The `perf` screen likewise replays its source 0% / 33% / 67% / 100% snapshots in one overwritten `Running[perf]` line and freezes the full source `performance_summary.txt` table verbatim, including its Basic Information, three-tier overview, all per-request metrics, and all workload-throughput rows; only the absolute local report path is redacted. The `service` screen shows its actual startup, Dashboard URL, curl command, health JSON and 200 line. The terminal carries an explicit `recorded replay` label.
- Tab buttons use native `role="tab"` / `role="tabpanel"` state, support click plus Left/Right/Home/End navigation, and auto-advance every 20 seconds until the user clicks, uses the keyboard, or copies a command. Hover does not stop the rotation, so the automatic replay remains observable. Progress snapshots overwrite one terminal line and block subsequent output until the captured 100% frame has been shown. `prefers-reduced-motion` starts in a static state. Command copy uses the same Clipboard API behavior as `CodeSnippet`; output is plain non-editable text.
- At `max-width: 720px`, long terminal rows remain horizontally scrollable inside the single terminal screen; no architecture board or chapter navigation rule is changed.

**Verification**

- `npm run fix`
- `npm run verify`
- `git diff --check`

Current result: passed. The 1200 px in-app-browser capture verifies that the Eval tables are rendered as contiguous native-terminal text blocks, including the complete GSM8K perf-table header and values.

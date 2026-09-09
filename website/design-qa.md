# Design QA

## References and scope

- Approved visual direction: `/Users/yunlin/.codex/generated_images/01a084e3-705f-72e3-97d2-a1fc101393cc/exec-7378f081-96a9-430f-9950-d7fcdbe001bf.png`
- Local prototype: `http://127.0.0.1:4321/evalscope/`
- Browser inspection: Codex in-app browser, 2026-09-09, on Home, Evaluation, Agent & Harness, Performance, Benchmarks, Visualization, Research, and Get Started.

## Earlier QA baseline

- Recompared the approved reference with the local Home capture: warm paper ground, ink display type, technical blue labels, gold measurement marks, registration line, fine ruler ticks, and dark evidence layer are present without restoring the removed decorative excess.
- Header and footer use the supplied `docs/en/_static/images/evalscope_icon.png` asset plus editable `EvalScope` text; no legacy wordmark image is displayed.
- Each evidence-led route has a distinct primary source: Evaluation uses the qwen-plus GSM8K result, Agent uses the real Details view, Performance uses Overview and Charts, Visualization uses those three distinct views, and pages without a genuine dashboard view use code or artifact evidence rather than a fabricated screenshot.
- Replaced the malformed inline terminal blocks with the reusable `CodeSnippet` component. It preserves command line breaks, labels the snippet, uses a macOS-style toolbar and exposes a working Copy control. The local browser click completed without navigation or console-visible failure.
- The Home heading now has a measured four-line rhythm and its terminal layer contains only the runnable API-first command, so it is no longer hidden by the real dashboard screenshot.
- Removed the Evaluation demo metric strip: the `qwen-plus` / GSM8K result remains confined to its real screenshot and caption rather than being presented as a product-wide claim.
- Rebuilt the Benchmark hero as a measurement orbit: gold trajectory rings, dotted crosshair axes, blue registration nodes, and a fine-tick ruler replace the generic background grid.
- Removed the Performance curve label collision and stripped explanatory output from the Performance and service terminal snippets; cards now show only copyable commands.
- The Get Started hero now carries a concise Manual/API versus AI/Skill route card, eliminating the unused right-side space while keeping the detailed four-step routes below.
- Browser review confirms the Agent loop, real evidence captions, drift-boundary labels, Evaluation modality/contract path, Benchmark registration orbit, Research claim boundary, and four bounded Get Started prompts render in the intended information order.
- `npm run verify` passed: Astro diagnostics, ESLint, Prettier, metadata-content validation (256 records), and static build of 16 locale routes plus 404.
- Responsive breakpoints preserve the compact mobile menu and collapse dense card/timeline layouts at 960px, 900px, and 620px. The current refinement adds explicit four-, five- and six-step timeline grids at those breakpoints.

## Latest annotation resolution (uncommitted)

- Home no longer uses the large generated evidence or API-route illustrations. The evidence section is now a compact four-record board; clicking or focusing a record highlights the selected Dataset, Prediction, Review or Report stage.
- Decorative measurement motion is now pointer-driven: hero grids, fine rulers, registration points and gold rules respond to horizontal mouse position and reset on pointer leave. No background/evidence loop runs autonomously; `prefers-reduced-motion` disables the transition layer.
- The PromptCard copy control explicitly restores ink-on-paper contrast inside every light card, including those placed in dark sections.
- Footer is a compact two-part layout: identity plus three primary links, followed by one balanced metadata line.
- `evalscope eval` appears once only, in the Home hero. The Evaluation dark API-call section, its duplicate command and unrelated configuration screenshot were removed. The command keeps the masked environment export and forwards `--api-key "$DASHSCOPE_API_KEY"`.
- Agent no longer includes the secondary Details crop. Agent, Benchmark and Visualization timelines use one actual connecting rail that passes through their centered numbered nodes; no separately positioned arrows remain.
- Performance now draws its load curve with a single SVG polyline through the three data-point centers. The section renders all six metrics promised by its heading: RPS, tok/s, Latency, TTFT, Concurrency and Success.
- Visualization now presents four distinct real artifact views: evaluation overview, configuration/score, Agent Details, and performance charts. Its dark evidence section now includes a four-part artifact rack instead of a mostly empty call-to-action.
- Get Started is AI-agent-first: manual/API route panels, manual quick start and installation matrix are removed. The remaining four jobs are selectable tabs with a single focused prompt panel.
- No model, dataset download, GPU job or external model API request was executed.

## Verification

- Local Codex in-app browser review confirmed Home has one `run-evaluation.sh`, a focusable/clickable evidence board and visible PromptCard copy control; Evaluation no longer renders a duplicate command; Agent no longer renders the secondary artifact crop; Performance reports six metric labels; Get Started renders one tab list with four selectable jobs and no manual route.
- `npm run verify` passed: Astro diagnostics, ESLint, Prettier, metadata-content validation (256 records), and static build of 17 routes including 404.
- `git diff --check` passed.

Current result: ready for user visual review; changes remain intentionally uncommitted and unpushed.

## Follow-up annotation resolution

- Footer navigation now uses an explicit column grid and fixed inter-link spacing; links no longer visually concatenate.
- Home and Evaluation evidence paths are now true controls with pressed state and live explanatory panels. Selecting a step changes the index, label and explanation rather than only changing the background.
- The two desktop evidence headings opt into a larger available line and a controlled single-line treatment; narrow viewports return to normal wrapping.
- Measurement-surface cards now use a small inline index, short gold rule and native icon instead of a boxed pseudo-instrument.
- The Agent loop keeps the connected rail and now includes a real Agent Trace artifact underneath. The Benchmark contribution path likewise includes a real Dashboard trend artifact rather than a fabricated diagram.
- The Performance summary card splits its values into three labelled statistic rows.
- The visualization gallery now uses real Dashboard Overview/trends, model comparison, prediction detail and Agent Trace screens sourced from the repository README/documentation references. The earlier configuration crop is removed.
- Catalog result count has a dedicated, readable result-status treatment instead of eyebrow-sized text.
- Hero sections now add a mouse-position-responsive radial registration glow and a subtle stack shift; the rest of the page retains the simpler pointer response.

## Current annotation resolution

- The evidence board now has distinct output targets. Selecting Dataset, Prediction, Review or Report only updates the lower live inspector; the tab's label and short description remain intact.
- Hero decoration now follows the supplied reference more closely: a localized paper-grid field, one subdued registration glow and sparse annotations remain, while the repeated gold bars and blue dots were removed from every content section.
- The Agent page uses a light real Agent result view in the loop section. Its metric grid now explains four generic evidence dimensions rather than presenting GSM8K-specific values as product claims.
- The Performance page replaces the isolated numeric-result panel with a decision checklist tied to its real overview and curve screenshots.
- The catalog updates both sides of its result count after a category, tag or search filter. Browser verification of `?category=Agent` showed `Showing 24 / 32 benchmarks` and only Agent records.
- The Benchmark contribution frame now uses the light real evaluation overview. The Get Started route connector extends continuously from step 01 to step 02.

## Verification refresh

- Codex in-app browser inspected Home, Evaluation, Agent, Performance and Benchmark routes. The Home and Evaluation controls retain their Dataset / Prediction / Review / Report labels and expose a separate live inspector; the Agent route has generic evidence language; the Performance route exposes the decision checklist; the filtered catalog renders the correct 24 / 32 result count.
- `npm run verify` passed after the final changes: Astro diagnostics, ESLint, Prettier, metadata-content validation (256 records), and static build of 17 routes including 404.
- `git diff --check` passed.

Final result: passed.

## Existing-evidence and reference-decoration refresh

- Reused the existing local `outputs/website-agent-trace` artifact; no evaluation was started for this refresh. Its Agent Trace has `python_exec` calls, a submitted answer of `18`, and a recorded score of `100%`.
- Captured four English/light Dashboard views without browser chrome: Dashboard history, Performance runs, the evaluation overview, and the Agent Trace. The Agent and Benchmark sections use these readable captures rather than a URL-bar crop; the Agent image shows tool calls and the submitted answer.
- Replaced the repeated decorative grid with the supplied measurement language across non-dark pages: localized ruler ticks, registration crosses and a low-contrast paper grid. They are pseudo-elements at `z-index: 0`; each content container is at `z-index: 1`, so the ornament cannot cover controls, text or screenshots.
- Widened the Home code and result stack while retaining its intended overlap; the vertical reference annotation remains behind that stack.
- The footer metadata is centered as one balanced line on desktop and stacks only on narrow screens.
- Catalog records derive a human-readable task type from benchmark metadata descriptions and normalize structured metric definitions. For example, AGIEval renders `Mixed (Multiple-Choice QA + Open-ended Math)` and `Answer accuracy`, instead of `LLM · Evaluation` and `Metrics`.

## Verification refresh

- In-app browser inspected Home, Evaluation, Agent and Visualization in English. Home has the wider evidence stack; Evaluation's decoration is behind its text and screenshot; Agent shows the new light trace; Visualization contains four light Dashboard screenshots with no URL chrome.
- The local Dashboard was confirmed in light English before capture: its header presents the `中文` language-switch button and the moon icon for switching to dark mode.
- Targeted service contract test passed after allowing the existing trace's `total_usage` field to be rendered: `tests/service/test_api_contracts.py -k 'prediction_contract_supports_messages_trace or json_response_validates'` (2 passed).
- `npm run verify` passed: Astro diagnostics, ESLint, Prettier, metadata-content validation (256 records), and static build of 17 routes including 404.
- `git diff --check` passed.

Final result: passed; changes remain uncommitted and unpushed.

## Homepage conversion and navigation correction

- Removed the Research chapter from Home and from the fixed chapter index. Research remains a standalone route in global navigation, leaving Home focused on shipped capabilities and conversion.
- Rewrote the Evaluation evidence heading as `One reproducible path from input to decision.` so it does not pre-announce the Benchmark catalog count.
- Added a transparent hover bridge between the Home link and its detail-page menu. The menu remains visible while the pointer travels to it, and keyboard focus can advance into its direct detail links.
- Rebuilt the Get Started composition: a full-width two-part introduction now sits above two equal action paths, one for the AI handoff prompt and one for command-line quickstart. This removes the former sparse-left, stacked-right imbalance while retaining both conversion routes.

## Verification refresh

- In-app browser inspection confirmed the focused Home menu exposes Evaluation, Agent & Harness, Performance and Visualization; keyboard navigation enters Evaluation directly. The Home index contains no Research item, and the Get Started section presents side-by-side AI and command-line paths without the former empty left column.
- `npm run verify` passed: Astro diagnostics, ESLint, Prettier, metadata-content validation (256 records), and static build of all 17 routes. `git diff --check` passed.

Final result: passed; changes remain uncommitted and unpushed.

## Homepage information-architecture reset

- Reframed Home as the long-form product overview in this deliberate order: Overview → Evaluate → Agent & Harness → Performance → Benchmarks → Visualization → Research → Get started. Research now precedes the conversion section in both document order and the active chapter index.
- The fixed chapter index is only an in-page orientation control. Its labels now use `Visualization` consistently, while every Home chapter includes a direct route to the corresponding full page.
- Replaced the header's click-to-open `Product` control with a `Home` link whose hover/focus menu exposes direct routes to the four full product pages. Header Benchmarks, Research and Use with AI now also use their standalone routes instead of Home anchors.
- Reworked Get started into two explicit paths: a guarded AI handoff prompt and a copyable command-line quickstart that installs the service extra, runs five API-backed GSM8K samples, and opens the local Dashboard. All copied task prompts now tell an agent to inspect the repository guidance first, use existing configuration and environment variables, preserve artifacts, and report missing prerequisites rather than inventing a result.
- Changed the volatile catalog callout to `250+` and shortened the Home Visualization headline. The four visualization frames use one 16:9 treatment with top-aligned cropping, removing the prior inconsistent card heights.

## Verification refresh

- In-app browser accessibility inspection confirmed the header Home, Benchmarks, Research and Use with AI links resolve to their intended standalone routes; the Home index reads `Visualization`, then `Research`, then `Get started`; the Home Agent, Performance, Benchmark, Visualization, Research and Get started chapters expose their corresponding detail links; the benchmark callout is `250+`; and the Get started chapter contains both the AI handoff and the copyable `quickstart.sh` command path.
- `npm run verify` passed: Astro diagnostics, ESLint, Prettier, metadata-content validation (256 records), and static build of all 17 routes. `git diff --check` passed.

Final result: passed; changes remain uncommitted and unpushed.

## Long-form homepage architecture

- The Home route is now the primary Product narrative: Overview → Evaluation → Agent & Harness → Performance → Benchmarks → Evidence views → Use with AI → Research. It reuses the existing real evidence assets rather than introducing another visual system.
- Added a compact desktop chapter index. It appears after the opening hero, follows the scroll position, marks the active section, and links directly to each section anchor. The content sections reserve a right gutter so the fixed index never covers copy, screenshots or controls; it is hidden below the desktop breakpoint.
- Header Product, Benchmarks, Research and Use with AI links now land on the corresponding Home anchors. The old detail routes remain available as direct, route-compatible deeper entries.
- New Home chapters replace the need to visit separate Product heroes for the primary narrative: Agent uses the existing light trace, Performance uses its existing light run view, Benchmark discovery is summarized as registry evidence, and Evidence views brings the four real Dashboard views together.

## Verification refresh

- In-app browser opened the long Home route and the direct `#agent` anchor. The Agent chapter lands below the sticky header, the directory appears in its reserved gutter, and `Agents` is highlighted without covering the trace image.
- `npm run verify` passed: Astro diagnostics, ESLint, Prettier, metadata-content validation (256 records), and static build of all 17 routes. `git diff --check` passed.

Final result: passed; changes remain uncommitted and unpushed.

## Copy and ornament tightening

- Removed the duplicate hero registration rows (`01`–`07`) across all routes. Each page now begins with its single blue eyebrow, so the route label and page number no longer compete with the section title.
- Replaced Research's procedural arrow headline with `A reproducible path for evaluating evaluator changes.` and rewrote the supporting copy and five workflow cards around a clear baseline → controlled variant → held-out evidence → audit → recorded decision sequence.
- Shared page backgrounds now contain only the quiet reference marks from the supplied direction: a cross, ruler ticks and fine grid. The longer Home-only reference rail remains confined to the Home composition; no copied annotation text appears on other routes.
- The outlined button hover/focus state now explicitly uses a blue surface with white text, including `Add a benchmark`.
- Removed the Home measurement band's top rule, reduced its height and pulled it upward into the hero so it reads as one continuous composition while retaining the lower separator for the facts row.

## Verification refresh

- In-app browser review confirmed Research's new protocol title and card copy, Benchmark's single hero eyebrow and text-free background marks, Visualization's single hero eyebrow, and the Home measurement band without its former top divider.
- `npm run verify` passed before the final CSS cleanup; the final cleanup was then checked with `npm run check` and `git diff --check` (Astro diagnostics, ESLint and Prettier all clean).

Final result: passed; changes remain uncommitted and unpushed.

## Follow-up sizing correction

- Reduced the Home evidence stack from the previous 50/50 grid to a reserved-rail layout. The code and result frame remain larger than the original version, but leave a fixed clear column before the right-side reference text and ruler.
- Browser recheck at the desktop viewport confirms `SAME PROMPTS / BETTER INSIGHTS`, the vertical ruler, `DATA / MODELS / PERFORMANCE / REAL PROGRESS`, and the lower cross/inscription remain fully readable beside the stack.
- At 900px and below, the desktop rail reservation is removed before the hero becomes a single-column layout, preserving the mobile width.

## Follow-up structure correction

- Home now uses a three-column desktop composition: copy and calls to action, a vertically separated command/result stack, and a normal-flow right measurement rail. The rail no longer relies on page-edge absolute positioning.
- The lower-left progress inscription is in the copy column below the calls to action, eliminating its former overlap with the buttons.
- The capability ruler separates its vertical ticks from the numeric labels; labels sit below the tick baseline with the paper ground behind them.
- Visualization's `Aggregate → sample → artifact.` heading has a compact desktop treatment that remains on one line, while narrow screens restore normal wrapping.

## Decoration consistency refresh

- Home's measurement ruler now uses a complete even progression: `0`, `50`, `100`, `200`, `300`, `400`.
- Removed the date-bearing caption from every product screenshot; image frames now identify the product view through their contextual section copy and image alternative text rather than a dated label.
- Every non-dark hero and content section now uses the same reference rail language: cross, ruler ticks, paper grid, and the supplied compact text annotation. These ornaments remain at `z-index: 0`, while containers stay at `z-index: 1`.
- Non-home heroes reserve a narrow right rail on desktop, so the shared ornament is visible without covering their screenshot or text. The rail scales down at tablet widths and is hidden on mobile.
- Home's research section uses the same low-contrast reference rail as the top of the page rather than a separate background treatment.

## Verification refresh

- In-app browser review at the desktop viewport confirmed the Home ruler labels do not overlap, screenshot captions contain no dates, Home's right rail remains clear of the code/result stack, and Evaluation's rail occupies reserved space to the right of content.
- `npm run verify` passed: Astro diagnostics, ESLint, Prettier, metadata-content validation (256 records), and static build of 17 routes including 404.
- `git diff --check` passed.

Final result: passed; changes remain uncommitted and unpushed.

# Frontend Design QA — 2026-07-16

Status: **PASS**

## Scope

- Consolidate the duplicated dataset score table and score visualization.
- Reduce color noise and improve information hierarchy in Evaluation Result.
- Remove Storybook and pixel-screenshot regression infrastructure.
- Recheck the affected report flow in the real local Web Console with real report data.

## Evidence

Source references:

- Dataset score duplication: `/var/folders/r4/j25n9tjd31xflxf_cc7lc09h0000gp/T/codex-clipboard-984b09b8-481c-4615-adbc-ffe972d9990a.png`
- Evaluation Result before the color pass: `/Users/yunlin/.codex/attachments/4b662017-7764-4a05-83f7-011b7a5d048b/image-2.png`

Verified implementation:

- URL: `http://127.0.0.1:5173/reports/20260715_191304%40%40qwen-plus%3A%3Aiquiz%2C%20general_mcq?root_path=%2FUsers%2Fyunlin%2FCode%2Feval-scope%2Foutputs`
- Real data: `qwen-plus`, `iquiz + general_mcq`, 3 samples, 75.0% aggregate score.
- Desktop browser viewport: default in-app Browser viewport (captured at 1275 × 717).
- Responsive browser viewport: 390 × 844.
- Temporary, deliberately untracked QA captures: `/tmp/evalscope-design-qa/`.

Screenshots are not stored in the repository because pixel-level visual regression was explicitly removed. This document records the states and findings without reintroducing screenshot baselines.

## Visual comparison

### Dataset Scores

Before:

- The same two values appeared in two separate cards.
- The second card consumed a full section while adding only ranking numbers and longer bars.
- Table and visualization used different score colors, weakening the relationship between the repeated values.

After:

- One `Dataset Scores` card is the single score-comparison surface.
- The sortable table keeps Dataset, Score, Samples, precise percentages, and an inline bar in the same row.
- One neutral accent color carries comparison; score-specific semantic colors remain in the summary cards where they communicate best/worst status.
- With three or more datasets, Table/Radar is a view switch in the same card. With one or two datasets, the redundant switch and chart are omitted.

Desktop result: pass. The two-dataset report uses one compact card and exposes both score bars and exact values without duplicate content.

Mobile result: pass after iteration. The Samples column collapses into a subtitle under the dataset name, the score bar and percentage remain visible, and the table no longer requires horizontal scrolling. Summary cards switch to one column so dataset names do not collapse into vertical text.

### Evaluation Result

Before:

- Three competing label colors made equally important fields look like unrelated categories.
- Both JSON detail blocks were expanded by default, so diagnostic data dominated the answer and score.
- Large color surfaces and syntax colors competed with the result itself.

After:

- Labels use the neutral text hierarchy; only the score badge uses semantic success/danger color.
- Extracted Answer, Expected Answer, and Score form a clear responsive three-column summary on desktop and one-column stack on mobile.
- Score Detail and Metadata are collapsed by default and expose `aria-expanded` state.
- Detail content remains available without taking over the initial scan path.

Desktop and 390 px responsive results: pass.

## Interaction checks

| Interaction | Real-browser result |
| --- | --- |
| Open report Overview | Pass |
| Dataset Scores renders only once | Pass |
| Sort Score ascending/descending | Pass; row order changed from `general_mcq, iquiz` to `iquiz, general_mcq` |
| Open Predictions | Pass |
| Score Detail default state | Pass; `aria-expanded=false` |
| Expand and collapse Score Detail | Pass |
| Rapid Predictions → Overview switch | Pass; no new warning/error log |
| Mobile navigation and report rendering | Pass |
| Mobile score values and sample counts | Pass; no horizontal clipping |
| Radar switch with >=3 datasets | Component test pass; the current real outputs contain at most two datasets, so no real report can expose this conditional control |

## Issues found and resolved during QA

| Priority | Finding | Resolution |
| --- | --- | --- |
| P1 | The first merged table still clipped percentage and Samples at 390 px. | Added responsive column presentation and compact score sizing; Samples moves below the dataset name on small screens. |
| P1 | Two-column summary cards forced dataset names into near-vertical text on a narrow report column. | Summary cards now use one column below the `sm` breakpoint. |
| P2 | Aborted subset/prediction requests were logged as errors during rapid tab changes or reloads. | Typed `DomainError(kind='aborted')` is now ignored before UI error state or console logging. |
| P2 | Storybook removal exposed stale type imports and an inappropriate production build dependency on test sources. | Page imports now use the co-located empty-state types; test sources are excluded from the production TypeScript build; Node scripts use automatic module detection. |

Unresolved P0/P1/P2 findings: **none**.

## Automated verification

- ESLint: pass.
- Vitest: 39 files, 219 tests passed.
- Design drift: CSS token source synchronized, structure check passed, locale keys consistent.
- Production TypeScript/Vite build: pass.
- Initial-load size budget: 118.70 KB gzip, below the 150 KB limit.
- Storybook reference scan across active package metadata, lockfile, frontend CI, and E2E README: no matches.

## Design decision

The optimized pattern is **one comparison card with progressive disclosure**, not two permanently visible representations. A table is the default because it supports exact values, samples, navigation, and sorting. Radar is optional only when the dataset count is large enough to produce a meaningful shape. This keeps the report compact without removing analytical capability.

## Dashboard 4K width QA

### Scope and decision

The dashboard keeps its existing single-column information architecture. The page content is centered and capped at 1280 px (`max-w-7xl`); no split-column layout, decorative expansion, or additional wide-screen-only feature was introduced.

### Visual evidence

- Reference capture: `/var/folders/r4/j25n9tjd31xflxf_cc7lc09h0000gp/T/codex-clipboard-ec60d18b-2256-475e-bfa3-565cb05e0787.png`.
- Combined before/after inspection: `/tmp/evalscope-dashboard-optimization/dashboard-before-after-wide.png`.
- Mobile implementation capture: `/tmp/evalscope-dashboard-optimization/mobile-390x844-raw.jpg`.

| Viewport | Measured content result | Overflow |
| --- | --- | --- |
| 3840 x 2160 | Dashboard width 1280 px; left and right margins 1280 px; both direct sections aligned to the same boundary | None |
| 2048 x 1152 | Dashboard width 1280 px; left and right margins 384 px | None |
| 390 x 844 | Run row height 71 px; model column 157.5 px after moving the timestamp into the mobile detail stack | None |

### Findings and resolutions

| Priority | Finding | Resolution |
| --- | --- | --- |
| P2 | On 4K and other wide displays, the dashboard stretched across the entire viewport, making the Recent Runs content feel sparse and disconnected. | Applied one page-level 1280 px maximum width and automatic side margins so the path control, summary cards, and Recent Runs share one centered boundary. |
| P2 | The desktop-oriented fixed timestamp column left too little room for model and dataset names on a 390 px viewport. | Switched the row to a responsive grid and moved the timestamp below the model details on small screens. |

### Real-browser interaction checks

| Interaction | Result |
| --- | --- |
| All filter | Pass; restored all 54 runs |
| Perf filter | Pass; displayed 3 performance runs |
| Search for `qwen3-max` | Pass; displayed 7 matching runs |
| Clear search | Pass; restored the complete feed |
| 4K layout measurement | Pass; centered 1280 px content and no horizontal overflow |
| 390 px layout measurement | Pass; no horizontal overflow |

### Verification

- ESLint: pass.
- Vitest: 39 files, 219 tests passed.
- Design drift checks: pass.
- Production build: pass.
- Initial-load size budget: 118.69 KB gzip, below the 150 KB limit.

final result: passed

## Evaluation list surface unification QA

### Scope and visual evidence

- Evaluations before/after comparison: `/tmp/evalscope-list-unification/evaluations-before-after.png`.
- Performance before/after comparison: `/tmp/evalscope-list-unification/performance-before-after.png`.
- Evaluations implementation capture: `/tmp/evalscope-list-unification/04-evaluations-after.png`.
- Performance implementation capture: `/tmp/evalscope-list-unification/03-performance-after.png`.

### Findings and resolutions

| Priority | Finding | Resolution |
| --- | --- | --- |
| P1 | Evaluations advertised comparison while hiding every selection checkbox until a separate mode button was activated. | Removed the explicit mode. Row and select-all checkboxes are always visible; row clicks still open details and checkbox clicks only update comparison selection. |
| P2 | Evaluations and Performance repeated their active navigation label as an otherwise empty breadcrumb row. | Removed both single-item breadcrumb rows. |
| P2 | The two primary list pages used different content widths, control ordering, card density, and row grouping. | Both pages now use the same centered 1280 px maximum width. Performance uses a search-first control row and one bordered, divided, column-aligned list surface matching Evaluations. |
| P2 | Dashboard Recent Runs placed model metadata near the left edge and the result at the far right, leaving a large unstructured gap. | Added the same Model, Dataset, Date, and Result column rhythm used by Evaluations while preserving the compact mobile layout. |

### Real-browser checks

| Interaction | Result |
| --- | --- |
| Select two evaluation rows | Pass; both checkboxes exposed `aria-checked=true` and the tray displayed `2 selected` |
| Open evaluation comparison | Pass; navigated with both selected report identifiers and loaded score comparison data |
| Select two performance rows | Pass; Compare changed from disabled `Compare (0)` to enabled `Compare (2)` |
| Open performance comparison | Pass; navigated with both paths and loaded the comparison report |
| Dashboard Perf filter | Pass; displayed three column-aligned performance rows |
| Horizontal overflow at the tested desktop viewport | None on Evaluations, Performance, or Dashboard |

### Automated verification

- ESLint: pass.
- Vitest: 39 files, 219 tests passed.
- Design drift checks: pass.
- Production build: pass.
- Initial-load size budget: 118.68 KB gzip, below the 150 KB limit.

final result: passed

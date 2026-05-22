---
name: EvalScope Dark Console
colors:
  # Brand & Accent
  accent: "#816DF8"
  accent-dark: "#5B3FD6"
  accent-dim: "rgba(129,109,248,0.12)"
  purple: "#a78bfa"

  # Surface ladder (sunken → elevated)
  bg: "#0c0c1a"
  bg-deep: "#09091a"
  bg-card: "#12122b"
  bg-card2: "#16163a"
  surface-glass: "rgba(18,18,43,0.7)"

  # Text (3-step ladder)
  text: "#e2e8f0"
  text-muted: "#8896aa"
  text-dim: "#505870"
  on-filled: "#ffffff"

  # Hairline borders
  border: "rgba(129,109,248,0.10)"
  border-md: "rgba(129,109,248,0.18)"
  border-strong: "rgba(129,109,248,0.28)"

  # Semantic states
  success: "#10b981"
  warning: "#f59e0b"
  danger: "#ef4444"
  info: "#60a5fa"

  # Deep saturated pass/fail (boolean badges)
  pass: "rgb(45,104,62)"
  fail: "rgb(151,31,44)"

  # Light-theme companions (used when [data-theme="light"])
  bg-light: "#f5f6fa"
  bg-card-light: "#ffffff"
  bg-card2-light: "#eef0f7"
  accent-light: "#6c57e8"
  text-light: "#1a1f2e"
  text-muted-light: "#5a6378"
  text-dim-light: "#9aa0b4"

  # Compare slot accents (per-model tagging in compare view)
  compare-0: "#818cf8"
  compare-1: "#34d399"
  compare-2: "#fbbf24"

  # Chat-bubble role accents
  bubble-user: "#818cf8"
  bubble-bot: "#34d399"
  bubble-tool: "#fbbf24"
  bubble-reasoning: "#34d399"
  bubble-system: "rgba(148,163,184,1)"
typography:
  display-xl:
    fontFamily: System Sans
    fontSize: 24px
    fontWeight: 700
    letterSpacing: -0.02em
    lineHeight: 1.2
  title-md:
    fontFamily: System Sans
    fontSize: 16px
    fontWeight: 700
    letterSpacing: normal
    lineHeight: 1.25
  body-sm:
    fontFamily: System Sans
    fontSize: 14px
    fontWeight: 400
    letterSpacing: normal
    lineHeight: 1.5
  body-sm-strong:
    fontFamily: System Sans
    fontSize: 14px
    fontWeight: 500
    letterSpacing: normal
    lineHeight: 1.5
  body-xs:
    fontFamily: System Sans
    fontSize: 12px
    fontWeight: 400
    letterSpacing: normal
    lineHeight: 1.5
  label-xs:
    fontFamily: System Sans
    fontSize: 12px
    fontWeight: 600
    letterSpacing: 0.05em
    textTransform: uppercase
    lineHeight: 1.4
  table-xs:
    fontFamily: System Sans
    fontSize: 10px
    fontWeight: 600
    letterSpacing: 0.05em
    textTransform: uppercase
    lineHeight: 1.4
  caption-mono:
    fontFamily: System Mono
    fontSize: 12px
    fontWeight: 400
    letterSpacing: normal
    lineHeight: 1.4
  code:
    fontFamily: System Mono
    fontSize: 13px
    fontWeight: 400
    letterSpacing: normal
    lineHeight: 1.5
  button-sm:
    fontFamily: System Sans
    fontSize: 12px
    fontWeight: 500
    letterSpacing: normal
    lineHeight: 1.4
  button-md:
    fontFamily: System Sans
    fontSize: 14px
    fontWeight: 500
    letterSpacing: normal
    lineHeight: 1.4
  button-lg:
    fontFamily: System Sans
    fontSize: 16px
    fontWeight: 500
    letterSpacing: normal
    lineHeight: 1.4
fontFamily:
  sans: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
  mono: '"SF Mono", "Cascadia Code", "Fira Code", monospace'
rounded:
  none: 0px
  xs: 4px
  sm: 8px
  md: 12px
  lg: 16px
  xl: 20px
  full: 9999px
spacing:
  xs: 4px
  sm: 8px
  md: 12px
  lg: 16px
  xl: 20px
  2xl: 24px
  3xl: 32px
  4xl: 48px
  5xl: 64px
shadows:
  sm: "0 2px 8px rgba(0,0,0,0.4)"
  md: "0 4px 20px rgba(0,0,0,0.55)"
  lg: "0 8px 40px rgba(0,0,0,0.6)"
  glow: "0 0 20px rgba(129,109,248,0.25)"
  glow-soft: "0 0 12px rgba(129,109,248,0.2)"
gradients:
  brand: "linear-gradient(135deg, #816DF8 0%, #a78bfa 100%)"
  accent: "linear-gradient(135deg, #0F9C7E 0%, #06b6d4 100%)"
  surface: "linear-gradient(135deg, rgba(129,109,248,0.08) 0%, rgba(167,139,250,0.05) 100%)"
  kpi-0: "linear-gradient(135deg, #6366f1, #8b5cf6)"
  kpi-1: "linear-gradient(135deg, #10b981, #06b6d4)"
  kpi-2: "linear-gradient(135deg, #f59e0b, #f97316)"
  kpi-3: "linear-gradient(135deg, #ec4899, #8b5cf6)"
  nav-hairline: "linear-gradient(90deg, transparent 0%, #816DF8 50%, transparent 100%)"
transition:
  fast: "150ms cubic-bezier(0.4, 0, 0.2, 1)"
  base: "180ms ease"
  slow: "400ms cubic-bezier(0.4, 0, 0.2, 1)"
breakpoints:
  sm: 640px
  md: 768px
  lg: 1024px
  xl: 1280px
container:
  max-width: 1600px
  page-padding-x: 16px
  page-padding-y: 20px
score-formula:
  foreground: "hsl(score * 120, 70%, 45%)"
  background: "hsl(score * 120, 70%, 45% / 0.15)"
  description: "0 → red, 0.5 → yellow, 1 → green — used for all dynamic score chips and badges"
---

# Design System: EvalScope Dark Console

## Overview

EvalScope's web dashboard is a developer-platform brand for **LLM evaluation and benchmarking** — the page is an instrument panel for engineers running evals, written for people who already know the syntax. It earns that posture with a stark dark-indigo system: near-black `{colors.bg}` canvas, ice-cool `{colors.text}` body, a 3-step text contrast ladder (`{colors.text}` → `{colors.text-muted}` → `{colors.text-dim}`) that gives every label, value, and metadata caption its own deliberate step. The only place the brand introduces saturation at console scale is the single brand violet `{colors.accent}` (`#816DF8`) — applied to active states, primary CTAs, focus rings, and the wordmark accent — and the *dynamic HSL score gradient* that maps a 0-1 metric to a `red → yellow → green` chip color. That score gradient is the entire emotional payload of the product: a benchmark either passes or fails, and the UI must say so at a glance.

Type is the second decisive voice. The brand uses **the native system stack** (no web font is loaded) — `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, ...` for narrative and `"SF Mono", "Cascadia Code", "Fira Code"` for technical labels. The intent is "feels native to your OS, never a marketing site." Headlines are sentence-case with `tracking-tight` on display numbers; **all-caps + `tracking-wider`** is reserved for tiny section eyebrows (12 px / 10 px), never for headlines. Weight ceiling for the sans is **700** (Tailwind `font-bold`); the typical working set is 400 / 500 / 600 / 700.

Surfaces use a four-step indigo ladder: `{colors.bg}` (the page itself), `{colors.bg-deep}` (sunken — inputs, nav pills), `{colors.bg-card}` (the default card surface), `{colors.bg-card2}` (hover and elevated rows). Shadows on dark are intentionally **deeper than they look** — `0 4px 20px rgba(0,0,0,0.55)` is the default card shadow — because near-black surfaces eat soft drops. Cards never float on a single heavy blur; they sit on the page held by a 1-px violet-tinted hairline + a tight stacked drop. A companion **light theme** flips polarity (paper-white page, white cards, `#6c57e8` accent) — the design assumes the user will toggle, but the dashboard is engineered dark-first.

**Key Characteristics:**

- A single violet primary CTA `{colors.accent}` carries every conversion target, paired with a transparent **ghost** secondary action. The brand uses a `{rounded.sm}` 8-px button shape for primary/secondary in the *console* (no marketing pills — this is an in-product surface).
- The primary CTA **glows on hover** (`box-shadow: 0 0 20px rgba(129,109,248,0.25)`). That violet glow is the brand's signature interaction.
- Every card section title, form label, and table header sets in `{typography.label-xs}` — 12 px (10 px for tables), `font-semibold`, **UPPERCASE**, `tracking-wider`, muted color. Body and titles stay sentence-case. The contrast between these two voices does most of the hierarchy work.
- A dynamic HSL **score chip** (`hsl(score × 120, 70%, 45%)`) is the second-most-recognizable component after the brand violet — it is how the product communicates pass/fail.
- The dashboard has both a **dark theme** (default, the natural state of the product) and a **light theme** (companion). Tokens are shared by name; only values flip. Theme is persisted to `localStorage` and applied via `data-theme` on `<html>` before first paint to avoid a flash.
- A complete domain token set exists for **chat bubbles** (5 semantic roles: user / bot / tool / reasoning / system), **compare slots** (3 per-model accent colors), and **KPI gradients** (4 named gradient pairs) — these are first-class brand tokens, not ad-hoc colors.
- A glassmorphic sticky top-nav (52 px, 12 px backdrop-blur, 1-px violet-to-transparent gradient hairline along the top edge) is the only "marketing-y" flourish the product allows itself.

## Colors

> **Note on dual theme.** Every color token below has a *dark* (default) and *light* value. Hex pairs are listed as `dark / light`. Components reference tokens by name — never by raw hex — so theme switching is free.

### Brand & Accent

- **Violet** (`{colors.accent}` — `#816DF8` / `#6c57e8`): The single brand color. Primary CTA fill, active nav fill, focus ring, brand-wordmark accent, table-header active-sort color. Used sparingly — should occupy under ~10 % of any screen.
- **Violet Deep** (`{colors.accent-dark}` — `#5B3FD6` / `#4f3ec8`): Hover state for primary CTAs.
- **Lavender** (`{colors.purple}` — `#a78bfa` / `#7c6be0`): The gradient companion stop; the second half of `{gradients.brand}`.
- **Violet Mist** (`{colors.accent-dim}` — `rgba(129,109,248,0.12)` / `rgba(108,87,232,0.10)`): The 8-12 % alpha violet used as pill background and focus-ring fill.
- **Violet Glow** (`{shadows.glow}` — `0 0 20px rgba(129,109,248,0.25)`): The signature hover halo on primary buttons and active nav.

### Surface

The brand operates with a 4-step indigo ladder, sunken-to-elevated:

- **Page** (`{colors.bg}` — `#0c0c1a` / `#f5f6fa`): The page body. Sets the dark instrument-panel mood.
- **Sunken** (`{colors.bg-deep}` — `#09091a` / `#ebebf5`): One step *below* the page — used for input wells, the deep-well that holds pill-style tab containers, and the icon tile in empty-states.
- **Card** (`{colors.bg-card}` — `#12122b` / `#ffffff`): The default card / dialog / table surface.
- **Card Elevated** (`{colors.bg-card2}` — `#16163a` / `#eef0f7`): Hover state for clickable cards and rows; also the inactive-tab fill in pill-tab containers.
- **Glass** (`{colors.surface-glass}` — `rgba(18,18,43,0.7)` / `rgba(255,255,255,0.80)`): Translucent indigo for the sticky top-nav, used with a 12-px backdrop-blur.

### Text

- **Ink** (`{colors.text}` — `#e2e8f0` / `#1a1f2e`): All headings, body, table cell values, button labels on non-filled surfaces.
- **Muted** (`{colors.text-muted}` — `#8896aa` / `#5a6378`): Secondary labels, nav-link inactive text, card-header micro-labels, button "ghost" idle text. *This is also the color section-eyebrow uppercase labels are set in.*
- **Dim** (`{colors.text-dim}` — `#505870` / `#9aa0b4`): Lowest-priority text — placeholder text, timestamps in compact rows, table empty-state. ⚠️ See *Do's and Don'ts* — this token sits near the AA contrast floor on dark cards (~3.5:1); reserve for ≥ 14 px non-essential metadata.
- **On Filled** (`{colors.on-filled}` — `#ffffff` / `#ffffff`): Text on `{colors.accent}` and other saturated fills.

### Semantic

- **Success** (`{colors.success}` — `#10b981` / `#059669`) / **Success Bg** (`rgba(16,185,129,0.08)`) / **Success Border** (`rgba(16,185,129,0.20)`): Confirmed / passed states; success toasts; chat-bot bubble border.
- **Warning** (`{colors.warning}` — `#f59e0b` / `#d97706`) / **Warning Bg** (`rgba(245,158,11,0.08)`) / **Warning Border** (`rgba(245,158,11,0.20)`): Pending / caution; tool-call chat bubbles.
- **Danger** (`{colors.danger}` — `#ef4444` / `#dc2626`) / **Danger Bg** (`rgba(239,68,68,0.08)`) / **Danger Border** (`rgba(239,68,68,0.20)`): Errors, fail badges, validation, destructive actions.
- **Info** (`{colors.info}` — `#60a5fa` / `#3b82f6`): Latency chart series, informational toasts.
- **Pass** (`{colors.pass}` — `rgb(45,104,62)` / `rgb(16,108,55)`) / **Fail** (`{colors.fail}` — `rgb(151,31,44)` / `rgb(180,30,42)`): Deep saturated greens / crimsons for boolean pass/fail badges where the tone needs more weight than the soft semantic family.

### Score Gradient (Signature)

The product's emotional core. A 0-1 score maps to **`hsl(score × 120, 70%, 45%)`**: 0 → red, 0.5 → yellow, 1 → green. Used as both foreground and translucent background on score chips, dataset chips, and group-header best-score callouts. This is computed inline (`scoreColor` / `scoreBg` helpers), never stored as a static palette. **Treat the formula as a brand asset** — do not reskin to a 5-step bucket, do not introduce a 4th hue.

### Compare Slots

Three per-model accent colors used to tag side-by-side model comparisons. Each slot has a `dot`, `border`, `bg`, and `bg-header` tint at ~10-30 % alpha:

- **Slot 0** (`{compare.0.dot}` — `#818cf8` / `#6c57e8`): Indigo.
- **Slot 1** (`{compare.1.dot}` — `#34d399` / `#0a8a6e`): Mint.
- **Slot 2** (`{compare.2.dot}` — `#fbbf24` / `#d97706`): Amber.

If a comparison view exceeds 3 models, *do not invent a 4th brand color* — collapse into a numbered legend instead.

### Chat Bubble Roles

Five semantic roles, each with a complete 7-token set (`bg`, `bg-hl`, `border`, `border-hl`, `icon-bg`, `icon-border`, `color`):

- **User** — Indigo family (`#818cf8` color).
- **Bot** — Emerald family (`#34d399` color).
- **Tool** — Amber family (`#fbbf24` color).
- **Reasoning** — Dim emerald (lower-saturation companion to Bot).
- **System** — Slate gray (`rgba(148,163,184,*)`).

Bubble containers are `{rounded.md}` with the role's tint background and border; hover/highlight states use the `*-hl` variants.

### KPI Gradients

Four named linear gradients for the four hero KPI tiles on the dashboard:

- **Indigo→Violet** (`{gradients.kpi-0}` — `linear-gradient(135deg, #6366f1, #8b5cf6)`)
- **Emerald→Teal** (`{gradients.kpi-1}` — `linear-gradient(135deg, #10b981, #06b6d4)`)
- **Amber→Orange** (`{gradients.kpi-2}` — `linear-gradient(135deg, #f59e0b, #f97316)`)
- **Pink→Violet** (`{gradients.kpi-3}` — `linear-gradient(135deg, #ec4899, #8b5cf6)`)

Always applied to the 40 × 40 `{rounded.md}` icon tile inside a `{components.kpi-card}`. **Same gradient values on both themes** — they're saturated enough to work over either canvas.

### Brand Gradients (Decorative)

- **Brand** (`{gradients.brand}` — `linear-gradient(135deg, #816DF8 → #a78bfa)`): For `gradient-text` and large brand moments.
- **Accent** (`{gradients.accent}` — `linear-gradient(135deg, #0F9C7E → #06b6d4)`): For the optional emerald-to-cyan accent text.
- **Surface** (`{gradients.surface}` — `linear-gradient(135deg, rgba(129,109,248,0.08) → rgba(167,139,250,0.05))`): The subtle violet wash layered behind KPI cards via `::before`.

## Typography

### Font Family

Two faces carry the entire system — **both are the user's OS-native faces** (no `@font-face` is loaded; this is deliberate):

1. **System sans** (`{typography.font-sans}` — `-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif`): Every display, body, button, link, and label. Working weights: 400 / 500 / 600 / 700.
2. **System mono** (`{typography.font-mono}` — `"SF Mono", "Cascadia Code", "Fira Code", monospace`): Timestamps, scores, model IDs, and any tabular-numeric data. Weight 400 only.

Antialiasing is forced (`-webkit-font-smoothing: antialiased`). No web font is loaded — the brand reads as "native developer tool, not marketing site" precisely because of this.

### Hierarchy

| Token | Size | Weight | Tracking | Use |
|---|---|---|---|---|
| `{typography.display-xl}` | 24px | 700 | tight | KPI value, hero number (dashboard `{components.kpi-card}`). |
| `{typography.title-md}` | 16px | 700 | normal | Card-title model name, group-header titles, brand wordmark. |
| `{typography.body-sm}` | 14px | 400 / 500 | normal | Default body text, button-md label, table-cell text, paragraph copy. |
| `{typography.label-xs}` | 12px | 600 | wider | **UPPERCASE** card-section header, form label, badge text — the brand's signature eyebrow. |
| `{typography.body-xs}` | 12px | 400 / 500 | normal | Empty-state hint, pill / badge body, mobile-nav link. |
| `{typography.table-xs}` | 10px | 600 | wider | **UPPERCASE** table-header micro-text — whispers, doesn't shout. |
| `{typography.caption-mono}` | 12px | 400 (mono) | normal | Timestamps, score values, dataset names in chips. |
| `{typography.code}` | 13–14px | 400 (mono) | normal | Log viewer, JSON viewer, terminal-style output. |
| `{typography.button-sm}` | 12px | 500 | normal | `{components.button}` `size="sm"`. |
| `{typography.button-md}` | 14px | 500 | normal | `{components.button}` `size="md"` (default). |
| `{typography.button-lg}` | 16px | 500 | normal | `{components.button}` `size="lg"` — for hero callouts only. |

### Principles

- **UPPERCASE + `tracking-wider` is the eyebrow voice — never the headline voice.** It marks "this introduces a region." Card titles, model names, and KPI labels stay sentence-case (or lower-case for the brand wordmark's lowercase "v").
- **`tracking-tight` is reserved for the display tier** — KPI numbers and the brand wordmark only. It tells the reader "this is set big on purpose."
- **`tabular-nums` + `font-mono` everywhere numbers must align** — KPI values, score chips, timestamps, percentage cells. Treat numbers as data, not prose.
- **Weight 700 is the display ceiling.** No `font-extrabold` / `font-black`. The brand reads as a calmer system because of this.
- **Line-heights inherit Tailwind defaults** (1.5 for body, 1.25 for headings). Don't override globally; rely on padding for vertical rhythm in tight stacks (cards, chips).
- **No web font.** The system stack is the system. Loading Inter or Geist on top would break the "native console" feel.

### Note on Font Substitutes

EvalScope uses the OS-native stack, so there are no proprietary faces to substitute. If a future skin needs to enforce a *single* face across all OSes for screenshot consistency:

- **Sans substitute** — *Inter* (400 / 500 / 600 / 700) is the closest stylistic match to the SF-on-macOS rendering; preserves the geometric / neutral character.
- **Mono substitute** — *JetBrains Mono* (400) at 12–14 px matches the technical voice well; *IBM Plex Mono* is a close second.

## Layout

### Spacing System

- **Base unit**: 4 px (Tailwind's default scale).
- **Tokens** (Tailwind-aligned):
  - `{spacing.xs}` 4 px · `{spacing.sm}` 8 px · `{spacing.md}` 12 px · `{spacing.lg}` 16 px · `{spacing.xl}` 20 px · `{spacing.2xl}` 24 px · `{spacing.3xl}` 32 px · `{spacing.4xl}` 48 px · `{spacing.5xl}` 64 px.
- **Page padding**: 16 px horizontal, 20 px vertical (`px-4 py-5`) at all breakpoints — *no responsive expansion*. The 1600 px container does the breathing for us.
- **Card interior padding**: 20 px body (`{spacing.xl}`), 12 px header strips (`{spacing.md} {spacing.xl}`).
- **Inline gap**: `{spacing.md}` (12 px) for component rows inside a card, `{spacing.xl}` (20 px) for inter-section gaps between major page blocks.
- **Pill / chip gap**: `{spacing.sm}` (6-8 px) — tight, scan-friendly. Pills are meant to wrap.

### Grid & Container

- **Max width**: `1600px` (`max-w-[1600px]`). Centered (`mx-auto`). Wide enough for 4-up KPI strip + side-by-side comparison; narrow enough that line-length never sprawls.
- **Column patterns**:
  - **KPI strip**: 4-up at `lg+`, 2-up below (`grid-cols-2 lg:grid-cols-4`).
  - **Compare view**: 2-3 columns at desktop, vertical stack on mobile.
  - **Eval timeline**: single column of full-width rows, no grid.
  - **Form pairs**: `grid-cols-1 md:grid-cols-2` for label-value pairs.
- **Gutters**: 16 px horizontal at all sizes.

### Whitespace Philosophy

Whitespace separates the *bands* — not the components inside a band. Section spacing is generous (`flex flex-col gap-5` → 20 px between major blocks); card interiors are tight (`gap-2` / `gap-3` between rows inside a card). The page reads as engineered — *large gaps + tight interior, never the other way around*. The dark page lets cards float visually without needing margin to assert themselves; the hairline border does the bordering work.

The brand's voice is **information-rich but uncluttered** — a typical dashboard packs a 4-tile KPI strip, a 30-row sortable eval list, search + sort + view-toggle controls, and the page still feels scannable because:
1. Tokens enforce a 3-step text hierarchy on every row.
2. UPPERCASE eyebrows visually section the layout without horizontal rules.
3. Score chips compress a percentage + benchmark name + color signal into 6 characters of mono.

### Responsive Strategy

#### Breakpoints (Tailwind defaults)

| Name | Width | Key Changes |
|---|---|---|
| Mobile | < 640px | Nav collapses to hamburger drawer; KPI grid drops to 2-up; all controls stack. |
| Small | 640–767px | GitHub icon and locale toggle appear in nav; KPI still 2-up. |
| Tablet | 768–1023px | Nav switches to **icon-only mode** with tooltips; main content full-width. |
| Desktop | 1024–1279px | Full pill-style nav with icon + label; KPI strip goes 4-up. |
| Wide | ≥ 1280px | Container holds at `max-w-[1600px]`; bands stretch but content centers. |

#### Touch Targets

The top-nav icon-only buttons (tablet) are 32 × 32 — *under the 44 × 44 WCAG floor*. This is a known compromise for the developer-tool density; on actual touch devices, hit areas are extended via padding. Primary buttons reach ~36 px tall in `md` size and ~44 px in `lg` — meet the floor at `lg`.

#### Collapsing Strategy

- **Nav**: Desktop = pill-row with icon + label; tablet = icon-only pills; mobile = logo + hamburger toggling a stacked drop-down (`max-height` animated, 300 ms).
- **KPI strip**: 4-up → 2-up below `lg`. Each tile keeps its `{rounded.md}` 12-px shape and 20-px padding.
- **Eval timeline**: Single column at all sizes — already a vertical list.
- **Forms**: Two-column label/value at `md+`, single column below.
- **Table**: Horizontal scroll wrapper preserves all columns rather than dropping them. The card-shell border keeps the scroll area framed.

#### Image / Icon Behavior

- **Iconography**: `lucide-react`, almost always 14–18 px, color inherits `currentColor`. The only non-Lucide mark is the brand SVG in the top-nav (a hand-drawn triangle with an amber check, rendered with `currentColor` so it follows theme).
- **No marketing imagery**: This is a console — no hero photo, no customer-logo strip, no illustrated empty-states. Empty states use a single Lucide icon in a 64 × 64 `{rounded.lg}` deep-well tile.
- **Charts**: Plotly-based, theme-aware — chart series colors are `{colors.chart-*}` tokens (latency / TTFT / TPOT / token).

## Elevation & Depth

| Level | Treatment | Use |
|---|---|---|
| **L0 — Flat** | No border, no shadow. | Page body, large empty regions, full-bleed dark sections. |
| **L1 — Hairline** | 1-px `{colors.border}` (violet at ~10-12 % alpha) only. | Default content cards (`{components.card}`), table chrome, form inputs. |
| **L2 — Lifted** | `{shadows.sm}` (`0 2px 8px rgba(0,0,0,0.4)` dark / `0 2px 8px rgba(0,0,0,0.07)` light) + L1 hairline. | KPI cards, path-bar at the top of the dashboard, anything that needs to read as "above the page." |
| **L3 — Floating** | `{shadows.md}` (`0 4px 20px rgba(0,0,0,0.55)` dark) + L1 hairline. | Default for the primary card variants when the page already has heavy chrome. |
| **L4 — Elevated** | `{shadows.lg}` (`0 8px 40px rgba(0,0,0,0.6)` dark) + `{colors.border-strong}` border. | Hover state for clickable cards, modal / dialog surfaces, dropdown menus. |
| **L5 — Glow** | `{shadows.glow}` (`0 0 20px rgba(129,109,248,0.25)`) on top of L1-L3. | The signature violet halo — primary button hover, active nav pill, sometimes active tab. |

**Brand rule**: dark surfaces require *deeper* shadows than light surfaces. The dark theme's `--shadow` is `rgba(0,0,0,0.55)` because lighter values disappear against `{colors.bg-card}`. Don't simply downscale a light-theme shadow into the dark theme.

### Decorative Depth

- **Backdrop-blur**: 12 px on `{colors.surface-glass}` for the sticky top-nav. This is the only blur effect in the system.
- **Hairline gradient line**: The top of the nav draws a 1-px `transparent → {colors.accent} → transparent` line at 40 % opacity. The closest the design comes to "decoration."
- **Card lift**: Hover lifts L1/L2 cards `-2px` (`{components.card-hover}`) or `-3px` (`{components.kpi-card}`), simultaneously upgrading their shadow ladder one step.
- **Gradient text**: `.gradient-text` and `.gradient-text-accent` utilities apply `{gradients.brand}` / `{gradients.accent}` to text via `background-clip: text`. Use sparingly — reserved for hero brand moments, never for body or table cells.

## Shapes

### Border Radius Scale

| Token | Value | Use |
|---|---|---|
| `{rounded.none}` | 0px | Full-bleed sections, table cell interiors. |
| `{rounded.xs}` | 4px | `{tokens.radius-xs}` — tightest inline pill (rarely used directly). |
| `{rounded.sm}` | 8px | `{tokens.radius-sm}` — buttons (sm/md), inputs, tabs, view-toggle buttons. Most in-product chrome lives here. |
| `{rounded.md}` | 12px | `{tokens.radius}` — **default card radius**. Cards, KPI tiles, group-header containers, large buttons. |
| `{rounded.lg}` | 16px | `{tokens.radius-lg}` — empty-state icon tile, KPI icon tile. |
| `{rounded.xl}` | 20px | `{tokens.radius-xl}` — reserved (used by occasional rounded-2xl Tailwind class on welcome / empty states). |
| `{rounded.full}` | 9999px | Badges, chips, score pills, mobile-nav icon buttons. |

**No pill (100-px) shape.** Unlike Vercel's marketing pill, EvalScope is an in-product surface; all CTAs use `{rounded.sm}` 8 px. Pills (`{rounded.full}`) are exclusively for *data* — badges, chips, score indicators.

### "Photography" — Iconography Geometry

- **Brand mark**: Hand-drawn SVG triangle with an amber check; rendered inline at 28 × 25 px with `currentColor` so it follows theme.
- **Lucide icons**: 14 px in dense lists, 16 px in nav, 18 px in form controls, 28 px in empty-state hero tiles. Stroke width 1.5–2.
- **KPI icon tile**: 40 × 40, `{rounded.md}`, filled with one of `{gradients.kpi-0..3}`, white icon centered.
- **Chart**: Plotly canvas, no rounded corners; sits inside a `{rounded.md}` card frame.

## Components

### Buttons

The brand operates with three button variants — *all in-product scale*, no marketing pill:

**`{components.button.primary}`** — the canonical violet CTA.
- Background `{colors.accent}`, text `{colors.on-filled}`, shape `{rounded.sm}` 8 px (md/sm) or `{rounded.md}` 12 px (lg). Hover adds `{shadows.glow}` violet halo + `bg → {colors.accent-dark}`. Press scales to 0.98. **The glow IS the brand interaction.**

**`{components.button.ghost}`** — the transparent secondary.
- Transparent background → `{colors.bg-card}` on hover → `{colors.bg-card2}` on active. Text `{colors.text}`. Used inside top-nav, toolbars, and inline icon-buttons.

**`{components.button.outline}`** — the hairline tertiary.
- Transparent background with 1-px `{colors.border-md}` border. Hover swaps both border and text to `{colors.accent}` (no fill). For "alternative action" CTAs.

**Sizes**:
- `sm` — 12-px text (`{typography.button-sm}`), 6/12 padding, `{rounded.sm}`, ~28 px tall.
- `md` — 14-px text (`{typography.button-md}`), 8/16 padding, `{rounded.sm}`, ~36 px tall *(default)*.
- `lg` — 16-px text (`{typography.button-lg}`), 10/24 padding, `{rounded.md}`, ~44 px tall.

Disabled = `opacity: 0.5` + `cursor: not-allowed`. Transitions use `{tokens.transition}` (0.18 s ease) consistently — never custom per button.

### Cards & Containers

**`{components.card}`** — the canonical card.
- Background `{colors.bg-card}`, text `{colors.text}`, padding 20 px (`{spacing.xl}`), shape `{rounded.md}` 12 px, border 1-px `{colors.border}`, shadow `{shadows.sm}` (L2). Optional **uppercase section header** in a 12 px / 20 px header strip with a bottom border. Collapsible variant rotates a chevron 90 °.

**`{components.card-hover}`** — utility applied to clickable cards.
- Adds `-2 px translateY` on hover, upgrades shadow to `{shadows.lg}` (L4), strengthens border to `{colors.border-strong}`. All transitions in `{tokens.transition}`.

**`{components.kpi-card}`** — the dashboard hero metric tile.
- Same chrome as `{components.card}` but layered with `{gradients.surface}` (subtle violet wash, 5-8 % alpha) via `::before`. Hosts a 40 × 40 `{rounded.md}` icon tile filled with `{gradients.kpi-0..3}`. Hover lifts `-3 px` and ramps to L4 shadow. Stagger-animated on first paint (60 ms steps).

**`{components.card-glass}`** — the glassmorphic surface used by the top-nav.
- `{colors.surface-glass}` background + 12-px backdrop-blur. Always combined with a 1-px hairline border. Reserve for sticky-positioned surfaces — diffuse blur is performance-sensitive.

**`{components.row-card}`** — borderless button styled as a list row (the eval-run rows on the dashboard).
- Background `{colors.bg-card}`, padding 16-20 px, `{rounded.md}`, 1-px `{colors.border}` border. Hover only changes border tint to `{colors.border-md}` — *no transform* — kept calm for dense lists.

### Inputs & Forms

**`{components.input}`** — the canonical text input.
- Background `{colors.bg-deep}` (one step *below* its container — inputs read as "wells"), 1-px `{colors.border}` border, text `{colors.text}` in `{typography.body-sm}`. Focus: border → `{colors.accent}` + 1-px ring in `{colors.accent-dim}`. **Soft halo, never harsh outline.** Padding 8/12, `{rounded.sm}` 8 px.
- Error state: border + ring swap to the danger family; 12 px `{colors.danger}` helper line below.

**`{components.label}`** — input label.
- 12 px, `font-medium`, **UPPERCASE**, `tracking-wider`, `{colors.text-muted}`, placed above the input with 6-px gap.

**`{components.select}`** / **`{components.search-input}`** — same chrome as `{components.input}`. The select uses the native `<select>` styled with the same class.

### Tabs

**`{components.tabs}`** — pill-container.
- An outer container with `{colors.bg-deep}` background and 1-px border, holding inline buttons. Active tab fills with `{colors.accent}` + white text + soft glow `0 0 12 px rgba(129,109,248,0.2)`; inactive uses `{colors.bg-card}` → `{colors.bg-card2}` on hover. Reads as a segmented control, not a tab strip.

### Navigation

**`{components.top-nav}`** — sticky top navigation.
- 52 px tall, `position: sticky`, `z-50`, `{components.card-glass}` chrome with 12-px blur, 1-px violet-to-transparent gradient line on the top edge. Container caps at 1600 px. Three modes:
  - **Desktop (lg+)** — pill links with icon + label; active = solid `{colors.accent}` + glow.
  - **Tablet (md–lg)** — 32 × 32 icon-only buttons with `title` tooltips; same active state.
  - **Mobile (< md)** — logo + hamburger toggling a stacked drop-down with animated `max-height` (300 ms ease-in-out).

**`{components.nav-link}`** — the pill-style nav button.
- Padding 6/12, `{rounded.sm}`, text `{colors.text-muted}` → `{colors.text}` on hover, active = `{colors.accent}` background + white text + violet glow shadow.

**`{components.icon-button}`** — circular ghost icon container in the nav.
- 32 × 32, `{rounded.sm}`, transparent background → `{colors.bg-card2}` on hover, icon inherits `{colors.text-muted}` → `{colors.text}`.

### Tables

**`{components.table}`** — sortable data table.
- Wrapped in `{components.card}` chrome (`{rounded.md}` border + bg) with `overflow-x-auto`.
- **Header cells**: `{typography.table-xs}` — 10 px, semibold, UPPERCASE, `tracking-wider`, `{colors.text-dim}`. They *whisper*. Sortable headers show a triple-state chevron (`ChevronsUpDown` idle, `ChevronUp/Down` active) and lift to `{colors.text}` on hover; active sort column turns `{colors.accent}`.
- **Row dividers**: 1-px `{colors.border}`. Clickable rows hover-fill `{colors.bg-card2}`.
- **Empty state**: Centered, dimmed "No data" cell — no illustration.

### Badges, Chips & Pills

**`{components.badge}`** — fully rounded inline pill.
- `{rounded.full}`, 8/2 padding, `{typography.body-xs}`. Four variants pair an 8-10 % alpha background with a saturated foreground: `default → accent`, `success → green`, `warning → yellow`, `danger → red`.

**`{components.filter-chip}`** — dismissable filter chip.
- Same pill shape as `{components.badge}`, with an optional 3.5 × 3.5 circular dismiss button (`X` icon).

**`{components.score-chip}`** — the signature dynamic-score pill.
- `{rounded.full}`, 8/2 padding, `{typography.caption-mono}`, **HSL-computed foreground and background** (`scoreColor` / `scoreBg`). Format: `"benchmark-name 87.3"` — the chip *is* the data point. Used in the eval timeline, dashboard tiles, and the leaderboard rows.

### Signature Components

**`{components.kpi-card}`** — see *Cards & Containers* above.

**`{components.eval-run-card}`** — a full-width borderless button-styled card row.
- Two rows of content: bold model name + colored score pill on the right; second row has a mono timestamp + a wrap of `{components.score-chip}` entries (one per benchmark). Hover only changes border tint (no lift).

**`{components.chat-bubble}`** — five-role chat surface.
- `{rounded.md}` container with role-specific bg / border / icon-bg / icon-border / color from the `{colors.bubble-*}` token family (user / bot / tool / reasoning / system). Hover strengthens border via `*-hl` variants.

**`{components.empty-state}`** — first-contact / no-data state.
- Vertical-center stack: 64 × 64 `{rounded.lg}` deep-well tile holding a 28-px Lucide icon (`{colors.accent}` for welcome states, `{colors.text-dim}` for empty/no-results), followed by a 2-line message — title in `{typography.body-sm}` `{colors.text}` (welcome) or `{colors.text-muted}` (empty), hint in `{typography.body-xs}` `{colors.text-dim}`.

**`{components.path-bar}`** — the dashboard's "scan this directory" input row.
- `{components.card}` chrome at L2, hosting an icon, an input, and a primary button — flex-row with 12-px gap.

**`{components.score-badge}`** — the bold percentage pill at the top of an eval row.
- `{rounded.full}`, 10/2 padding, `{typography.body-sm}` bold + `tabular-nums`, HSL-computed fg/bg. Distinct from `{components.score-chip}` by size and weight.

### Examples (illustrative)

> These `ex-*` surfaces mirror the brand-native primitives for downstream consumers (kits, mockups, Stitch generation). Each references existing components so a re-skin re-skins all surfaces consistently.

**`ex-metric-tile`** — Dashboard KPI tile. Re-uses `{components.kpi-card}` with gradient icon tile + tabular value + uppercase label.
- Properties: `backgroundColor`, `iconGradient`, `rounded`, `padding`, `valueTypography`, `labelTypography`.

**`ex-eval-run-row`** — A row in the eval timeline. Re-uses `{components.eval-run-card}` chrome.
- Properties: `backgroundColor`, `borderColor`, `rounded`, `padding`, `scoreColor`, `chipColor`.

**`ex-score-chip`** — Dynamic HSL score pill. Re-uses `{components.score-chip}`.
- Properties: `score (0-1)`, `label`, computed `bg` and `fg` via `scoreColor`.

**`ex-model-group-header`** — Collapsible header for the grouped-by-model dashboard view. Re-uses `{components.row-card}` chrome with a chevron, model name, run-count, and best-score callout.
- Properties: `backgroundColor`, `borderColor`, `rounded`, `expanded`.

**`ex-leaderboard-row`** — A row in a benchmark leaderboard. Re-uses `{components.table}` row chrome.
- Properties: `rowBackground`, `borderColor`, `cellTypography`, `scoreChip`.

**`ex-chat-bubble-user`** / **`ex-chat-bubble-bot`** / **`ex-chat-bubble-tool`** — Three of the five chat-bubble roles. Re-use `{components.chat-bubble}` with the role's token family.
- Properties: `backgroundColor`, `borderColor`, `iconBackground`, `color`, `rounded`, `padding`.

**`ex-empty-state-card`** — No-data state. Re-uses `{components.empty-state}` inside a `{components.card}` shell.
- Properties: `backgroundColor`, `iconColor`, `messageTypography`.

**`ex-skeleton-row`** — Loading state for a table or list row.
- Properties: `backgroundColor`, `pulseAnimation` (`skeletonPulse`, 1.5 s).

**`ex-form-field`** — Label + input + optional error helper. Re-uses `{components.label}` + `{components.input}`.
- Properties: `labelTypography`, `inputBackground`, `borderColor`, `focusRingColor`, `errorState`.

**`ex-compare-column`** — A model column in the compare view, using one of the three `{compare.0..2}` slot colors.
- Properties: `dotColor`, `borderColor`, `headerBackground`.

## Do's and Don'ts

### Do

- Reserve `{colors.accent}` (`#816DF8`) for primary CTAs, active states, focus rings, and the wordmark accent. Brand violet IS the conversion target — keep it under ~10 % of any screen.
- Use `{rounded.sm}` 8 px for buttons / inputs / tabs, `{rounded.md}` 12 px for cards, `{rounded.full}` only for badges / chips / score pills. Each shape signals its category.
- Set every section eyebrow, form label, and table header in `{typography.label-xs}` — UPPERCASE + `tracking-wider`. This is the brand's hierarchy signal; without it, the design flattens.
- Use `{typography.caption-mono}` + `tabular-nums` for any numeric column — scores, timestamps, percentages. Treat numbers as data, not prose.
- Compute score colors with `hsl(score × 120, 70%, 45%)` and use the same formula for foreground and a translucent background. The dynamic chip is the product's emotional signal.
- Layer stacked shadows (a deep multi-stop shadow + a 1-px violet hairline) rather than single heavy drops. Cards sit on the page, not above it.
- Cycle page chrome through the indigo ladder `{colors.bg}` → `{colors.bg-deep}` → `{colors.bg-card}` → `{colors.bg-card2}`. Inputs sit *deeper* than cards; hover sits *higher*.
- Animate page transitions with `fadeInUp` (12 px translate + opacity, 400 ms ease-out) and stagger children at 60 ms. The motion is subtle — don't lengthen it.
- Persist the `data-theme` to `localStorage` and apply it pre-paint in `index.html` to avoid FOUC. Tokens are theme-agnostic by name; values flip.

### Don't

- Don't introduce a 6th brand hue or a 4th compare-slot color. The palette is closed at violet + emerald + amber + red + slate (plus the dynamic HSL score). New accents flatten the voice.
- Don't render headlines in all-caps. UPPERCASE is the eyebrow voice (12 px / 10 px micro-labels) — never the title voice. Card titles and model names stay sentence-case.
- Don't promote the sans to `font-extrabold` / `font-black`. The display weight ceiling is **700**.
- Don't use `{colors.text-dim}` for essential UI text on dark cards — its ~3.5:1 contrast against `{colors.bg-card}` is below WCAG AA for normal body. Reserve for ≥ 14 px non-essential metadata (timestamps, "empty" labels, scrollbar thumb).
- Don't drop a single heavy 8-px-blur drop-shadow on a card. The dark theme requires *deeper* multi-stop shadows (`rgba(0,0,0,0.55)` at 20-40 px) — soft drops disappear on near-black.
- Don't apply `{gradients.brand}` to body text or table cells. Gradient-text is for hero / wordmark moments only.
- Don't bypass `{components.button}` to write custom `bg-[var(--accent)]` buttons inline. The button variants encode the glow, the scale-press, and the disabled state — re-deriving them by hand drifts the brand.
- Don't use inline `style={{ background: 'var(--xxx)' }}` when a Tailwind class or the `formStyles` helper would do. Inline styles bypass the token abstraction and break theme switching for the hover state.
- Don't pair `{rounded.full}` pill shapes with `{rounded.md}` cards as siblings inside the same control group — pills are for *data*, sm/md radii are for *interactive containers*. Mixing them on the same row breaks the shape grammar.
- Don't ignore `prefers-color-scheme` on first visit. If the user has never toggled, fall back to the OS preference before defaulting to dark.
- Don't widen `{spacing.xl}` (20 px) section gaps past `{spacing.2xl}` (24 px). The product is information-dense by design; extra whitespace makes the dashboard feel half-empty rather than airy.

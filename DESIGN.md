---
name: EvalScope Console
colors:
  # Brand & Accent — IDENTICAL across both themes. Violet is the brand constant.
  accent: "#816DF8"
  accent-dark: "#5B3FD6"
  accent-dim: "rgba(129, 109, 248, 0.12)"
  purple: "#a78bfa"

  # Surface ladder (sunken → elevated) — DARK
  bg: "#0c0c1a"
  bg-deep: "#09091a"
  bg-card: "#12122b"
  bg-card2: "#16163a"
  surface-glass: "rgba(18, 18, 43, 0.7)"

  # Text (3-step ladder) — DARK
  text: "#e2e8f0"
  text-muted: "#8896aa"
  text-dim: "#7a8195"
  on-filled: "#ffffff"

  # Hairline borders — DARK (translucent violet, the near-black bg lets even 10% read)
  border: "rgba(129, 109, 248, 0.10)"
  border-md: "rgba(129, 109, 248, 0.18)"
  border-strong: "rgba(129, 109, 248, 0.28)"

  # Semantic states
  success: "#10b981"
  warning: "#f59e0b"
  danger: "#ef4444"
  info: "#60a5fa"

  # Deep saturated pass/fail (boolean badges)
  pass: "rgb(45,104,62)"
  fail: "rgb(151,31,44)"

  # ────────────────────────────────────────────────────────────────
  # Light theme — warm-cream Console (used when [data-theme="light"])
  # Distinct palette philosophy: warm-neutral surfaces, solid warm-grey
  # hairlines (NOT translucent violet), warm ink text. Same violet accent.
  # ────────────────────────────────────────────────────────────────

  # Surface ladder — warm-cream, sunken → elevated
  bg-light: "#faf9f5" # warm cream canvas — was cool #f5f6fa
  bg-deep-light: "#f0ebe1" # input wells, one step below canvas — was cool #e8eaf2
  bg-card-light: "#ffffff" # pure white — strongest possible contrast against cream canvas
  bg-card2-light: "#f5f0e7" # hover / elevated — warm cream-soft, was cool #eef0f7
  surface-glass-light: "rgba(250, 249, 245, 0.80)" # warm-tinted glass — was pure white

  # Accent (unchanged from dark — violet is the brand constant)
  accent-light: "#6c57e8"
  accent-dim-light: "rgba(108, 87, 232, 0.14)" # slightly stronger on white card

  # Text — warm-ink ladder
  text-light: "#141413" # warm near-black — was cool #1a1f2e
  text-muted-light: "#6c6a64" # warm grey — was cool #5a6378
  text-dim-light: "#8e8b82" # warm grey — was cool #7c8497

  # Hairlines — SOLID warm hex, not translucent violet. Three concrete tones.
  # Critical: translucent violet at 0.20 alpha composites to near-invisible
  # on white cards. Solid warm-grey gives every card a definite boundary.
  border-light: "#e6dfd8" # standard hairline — was rgba(violet,0.20)
  border-md-light: "#d6cdbe" # emphasized — was rgba(violet,0.30)
  border-strong-light: "#c1b6a3" # hover / focus boundary — was rgba(violet,0.40)

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
    fontSize: 12px
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
  sans: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif'
  mono: 'ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", "DejaVu Sans Mono", "Courier New", monospace'
rounded:
  none: 0px
  xs: "4px"
  sm: "8px"
  md: "12px"
  lg: "16px"
  xl: "20px"
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
  # Dark theme — single deep drop (works on near-black surfaces).
  sm: "0 2px 8px rgba(0, 0, 0, 0.4)"
  md: "0 4px 20px rgba(0, 0, 0, 0.55)"
  lg: "0 8px 40px rgba(0, 0, 0, 0.6)"
  glow: "0 0 20px rgba(129, 109, 248, 0.25)"
  glow-soft: "0 0 12px rgba(129, 109, 248, 0.20)"
  # Light theme — two-stop stacks tinted with warm-ink (matches text colour),
  # not slate. Slate-tinted drops on cream read as a cool-grey smudge and break
  # the warm canvas. Warm-ink stays consistent with the rest of the palette.
  sm-light: "0 1px 2px rgba(20, 20, 19, 0.04), 0 4px 12px rgba(20, 20, 19, 0.06)"
  md-light: "0 4px 16px rgba(20, 20, 19, 0.07), 0 12px 32px rgba(20, 20, 19, 0.05)"
  lg-light: "0 12px 24px rgba(20, 20, 19, 0.09), 0 24px 48px rgba(20, 20, 19, 0.07)"
  glow-light: "0 0 20px rgba(108, 87, 232, 0.22)"
  glow-soft-light: "0 0 12px rgba(108, 87, 232, 0.18)"
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
  base: "250ms cubic-bezier(0.4, 0, 0.2, 1)"
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
  foreground: "hsl(score * 120, var(--score-fg-s), var(--score-fg-l))  # dark: 70%/45%, light: 85%/32%"
  background: "RGB-interpolated translucent companion, alpha * var(--score-bg-a-mul)  # dark: ×1, light: ×1.6"
  description: "0 → red, 0.5 → yellow, 1 → green — used for all dynamic score chips, badges, and SVG rings. Saturation, lightness, and bg alpha multiplier are theme-scoped CSS vars so yellow mid-tones stay legible on warm-cream without re-coding the brand HSL. Light theme bumps saturation +15pt to compensate for the lightness compression at hue ≈ 60 (olive)."
---

# Design System: EvalScope Console

## Principles {#principles}

> **Addressable section — `principles`.** The design philosophy, brand posture, and dual-theme parity rules that govern every downstream decision. See also the normative *Do's and Don'ts* under [Decision Records](#decisions).

EvalScope's web dashboard is a developer-platform brand for **LLM evaluation and benchmarking** — the page is an instrument panel for engineers running evals, written for people who already know the syntax. It earns that posture through **two equally weighted themes** rather than one canonical mode with a translated companion. Both themes share the same vocabulary — same type, same spacing, same radii, same components — but each carries its own surface philosophy. They are two voices of one brand, not one design re-tinted.

**Dark Console** (default) is a stark dark-indigo system: near-black `{colors.bg}` canvas, ice-cool `{colors.text}` body, a 3-step text ladder. Hairlines are translucent violet at 10-28 % alpha because the near-black bg lets even a faint violet tint read as a definite edge. Shadows are single deep drops at `rgba(0,0,0,0.55)` — near-black eats soft shadows. The mood is *late-night terminal session*: low-light, low-distraction, the violet glow on a primary CTA is the only saturated thing on screen.

**Warm Console** (light theme) is a warm-cream system: cream `{colors.bg-light}` canvas (`#faf9f5`), pure white `{colors.bg-card-light}` cards, warm-ink `{colors.text-light}` body (`#141413`). Hairlines are **solid warm-grey** (`{colors.border-light}` `#e6dfd8`, `#d6cdbe`, `#c1b6a3`) — *not* translucent violet, because violet at 20 % alpha disappears against a white card and leaves every card boundary undefined. Shadows are two-stop stacks tinted with the same warm-ink as the text, so cards lift off the cream without printing a cool slate smudge. The mood is *morning code review*: high-contrast, restful on the eyes, the violet CTA is the only cool note in an otherwise warm palette.

The brand constant across both themes is the single violet `{colors.accent}` (`#816DF8` dark / `#6c57e8` light) used for primary CTAs, active nav states, focus rings, and the wordmark accent — plus the dynamic HSL score gradient (`hsl(score × 120, 70%, 45%)`) that maps a 0-1 metric to red → yellow → green. Both signals work over either canvas. Everything else — surface ladder, hairline material, shadow tint, on-canvas text colour — is theme-specific by design, because dark and light surfaces need *different* materials to produce the same hierarchy.

Type is the second decisive voice and is **theme-agnostic**. The brand uses cross-platform system font stacks (no web font is loaded) — `system-ui, -apple-system, "Segoe UI", Roboto, ...` for narrative and `ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", ...` for technical labels. Each OS resolves to its own native UI face. Headlines are sentence-case with `tracking-tight` on display numbers; **all-caps + `tracking-wider`** is reserved for 12 px section eyebrows and table labels, never headlines. Weight ceiling is **700**; the working set is 400 / 500 / 600 / 700.

**Key Characteristics:**

- **Dual-theme parity, not dual-theme translation.** Dark uses translucent violet hairlines on near-black; light uses solid warm-grey hairlines on cream. They produce the same hierarchy through opposite material choices. Theme is persisted to `localStorage` and applied via `data-theme` on `<html>` before first paint to avoid a flash.
- A single violet primary CTA `{colors.accent}` carries every conversion target on both themes, paired with a transparent **ghost** secondary. The brand uses a `{rounded.sm}` 8-px button shape for primary/secondary in the *console* (no marketing pills — this is an in-product surface).
- The primary CTA **glows on hover** with violet at 20-25 % alpha on both themes. That violet glow is the brand's signature interaction — identical animation, identical colour, identical timing across themes.
- Every card section title, form label, and table header is at least 12 px, `font-semibold`, **UPPERCASE**, `tracking-wider`, and uses the AA-safe muted color. Body and titles stay sentence-case. The contrast between these two voices does most of the hierarchy work.
- A dynamic HSL **score chip** (`hsl(score × 120, 70%, 45%)`) is the second-most-recognizable component after the brand violet — it is how the product communicates pass/fail. Identical formula on both themes; the chip's saturation works over cream and over near-black.
- **Light theme uses solid hex hairlines, not translucent violet.** This is the most important light-theme rule, and the one most often broken on first attempt: a violet alpha overlay disappears into a near-white page, so light surfaces require concrete warm-grey edges (`#e6dfd8` / `#d6cdbe` / `#c1b6a3`) to keep their boundaries.
- A complete domain token set exists for **chat bubbles** (5 semantic roles: user / bot / tool / reasoning / system), **compare slots** (3 per-model accent colors), and **KPI gradients** (4 named gradient pairs) — these are first-class brand tokens, not ad-hoc colors.
- A glassmorphic sticky top-nav (52 px, 12 px backdrop-blur, 1-px violet-to-transparent gradient hairline along the top edge) is the only "marketing-y" flourish the product allows itself. Dark uses translucent indigo glass; light uses translucent cream glass.

## Design Tokens {#tokens}

> **Addressable section — `tokens`.** The executable token reference: color, typography, layout, elevation, and shape scales. Runtime CSS custom properties in `evalscope/web/src/index.css` are the source of truth. `npm run design:tokens` synchronizes the matching frontmatter values, while the drift check fails CI when the generated documentation is stale.

### Colors

> **Note on dual theme.** Every color token below has a *dark* (default) and *light* value. Hex pairs are listed as `dark / light`. Components reference tokens by name — never by raw hex — so theme switching is free.
>
> **Light values use SOLID hex for hairlines, not translucent violet.** This is the structural difference from earlier light-theme generations and the single most-broken light-theme rule. See `{colors.border-light}` below and *Elevation & Depth* for the reason.

#### Brand & Accent

The accent family is **identical in spirit on both themes** — a single violet handles every conversion target. Slightly different hex values per theme (the dark violet is brighter to read on near-black; the light violet is a half-step deeper to hold weight against white cards) but the same brand voltage.

- **Violet** (`{colors.accent}` — `#816DF8` / `#6c57e8`): The single brand color. Primary CTA fill, active nav fill, focus ring, brand-wordmark accent, table-header active-sort color. Used sparingly — should occupy under ~10 % of any screen.
- **Violet Deep** (`{colors.accent-dark}` — `#5B3FD6` / `#4f3ec8`): Hover state for primary CTAs.
- **Lavender** (`{colors.purple}` — `#a78bfa` / `#7c6be0`): The gradient companion stop; the second half of `{gradients.brand}`.
- **Violet Mist** (`{colors.accent-dim}` — `rgba(129,109,248,0.12)` / `rgba(108,87,232,0.14)`): The low-alpha violet used as pill background and focus-ring fill. Light theme runs slightly stronger (0.14 vs 0.12) because white cards need a touch more saturation to read the mist.
- **Violet Glow** (`{shadows.glow}` — `0 0 20px rgba(129,109,248,0.25)` / `0 0 20px rgba(108,87,232,0.22)`): The signature hover halo on primary buttons and active nav. Same effect, same magnitude, on both themes.

#### Surface

Each theme operates with a 4-step surface ladder, sunken-to-elevated. The **ladder structure is shared**; the **material is different** — dark walks an indigo ladder, light walks a warm-cream ladder. The semantic of each step is the same: `bg-deep` is *below* the page, `bg-card` is the working surface, `bg-card2` is the elevated state.

- **Page** (`{colors.bg}` — `#0c0c1a` / `#faf9f5`): The page body. Dark = near-black instrument-panel mood. Light = warm cream canvas (Claude-style), deliberately not cool grey-white — the cool variant (`#f5f6fa`) reads as "any other SaaS dashboard" and washes out white cards.
- **Sunken** (`{colors.bg-deep}` — `#09091a` / `#f0ebe1`): One step *below* the page — used for input wells, the deep-well that holds pill-style tab containers, and the icon tile in empty-states. Dark goes deeper-black; light goes warmer-cream. On both themes, inputs read as "wells" because they're one step below the surface that contains them.
- **Card** (`{colors.bg-card}` — `#12122b` / `#ffffff`): The default card / dialog / table surface. Dark = indigo card on near-black page. Light = pure white card on cream canvas — the cream-to-white contrast (ΔL ≈ 7) is what gives a light-theme card its lift, NOT a heavy shadow.
- **Card Elevated** (`{colors.bg-card2}` — `#16163a` / `#f5f0e7`): Hover state for clickable cards and rows; also the inactive-tab fill in pill-tab containers. Light theme's elevated state is warm-cream-soft — the elevated state is darker on dark theme but lighter-than-card-but-warmer on light theme (the white card with a soft-cream hover reads as "depressed into the cream canvas").
- **Glass** (`{colors.surface-glass}` — `rgba(18,18,43,0.7)` / `rgba(250,249,245,0.80)`): Translucent surface for the sticky top-nav, used with a 12-px backdrop-blur. Light theme uses tinted cream glass (matches the canvas), NOT pure white — white glass on cream reads as a foreign sheet floating in space.

#### Text

- **Ink** (`{colors.text}` — `#e2e8f0` / `#141413`): All headings, body, table cell values, button labels on non-filled surfaces. Light theme uses warm-near-black (`#141413`, ≈ the same value Claude.com uses) rather than a cool slate (`#1a1f2e`), so the text temperature matches the canvas temperature.
- **Muted** (`{colors.text-muted}` — `#8896aa` / `#6c6a64`): Secondary labels, nav-link inactive text, card-header micro-labels, button "ghost" idle text. *This is also the color section-eyebrow uppercase labels are set in.* Light theme uses warm-grey (`#6c6a64`) rather than cool-slate (`#5a6378`) to stay coherent with the warm canvas.
- **Dim** (`{colors.text-dim}` — `#7a8195` / `#8e8b82`): Lowest-priority text — placeholder text, timestamps in compact rows, table empty-state. **Contrast tuned to ~3.6 : 1** against `{colors.bg-card}` on both themes — sits just above the WCAG AA Large floor (3 : 1), still **below AA Normal (4.5 : 1)**. ⚠️ Reserve for ≥ 14 px non-essential metadata. Light theme uses a warm-grey at the same luminance step as the dark theme's cool-grey — the perceived hierarchy stays identical.
- **On Filled** (`{colors.on-filled}` — `#ffffff` / `#ffffff`): Text on `{colors.accent}` and other saturated fills. Identical on both themes — the violet CTA is dark enough on both that white text holds.

#### Hairlines (the structural difference between themes)

- **Border** (`{colors.border}` — `rgba(129,109,248,0.10)` / `#e6dfd8`): The default 1-px card / input / divider boundary. **Dark uses translucent violet at 10 % alpha** because the near-black bg-to-card luminance step already does most of the boundary work — the violet hairline just tints it. **Light uses a SOLID warm-grey hex** (`#e6dfd8`, Claude-style cream-hairline) because the white-card-on-cream luminance step is gentle enough that a translucent violet overlay disappears into the page. Borders on light theme are concrete materials, not tints.
- **Border Emphasized** (`{colors.border-md}` — `rgba(129,109,248,0.18)` / `#d6cdbe`): One step stronger — used on form inputs after focus, on the active-state of hover cards, on the boundary between a card and a nested section.
- **Border Strong** (`{colors.border-strong}` — `rgba(129,109,248,0.28)` / `#c1b6a3`): The strongest boundary — used by `{components.card-hover}` on hover lift, and by elevated cards in modal contexts. On both themes this is the "this thing is grabbing attention" hairline.

#### Semantic

- **Success** (`{colors.success}` — `#10b981` / `#059669`) / **Success Bg** (`rgba(16,185,129,0.08)`) / **Success Border** (`rgba(16,185,129,0.20)`): Confirmed / passed states; success toasts; chat-bot bubble border.
- **Warning** (`{colors.warning}` — `#f59e0b` / `#d97706`) / **Warning Bg** (`rgba(245,158,11,0.08)`) / **Warning Border** (`rgba(245,158,11,0.20)`): Pending / caution; tool-call chat bubbles.
- **Danger** (`{colors.danger}` — `#ef4444` / `#dc2626`) / **Danger Bg** (`rgba(239,68,68,0.08)`) / **Danger Border** (`rgba(239,68,68,0.20)`): Errors, fail badges, validation, destructive actions.
- **Info** (`{colors.info}` — `#60a5fa` / `#3b82f6`): Latency chart series, informational toasts.
- **Pass** (`{colors.pass}` — `rgb(45,104,62)` / `rgb(16,108,55)`) / **Fail** (`{colors.fail}` — `rgb(151,31,44)` / `rgb(180,30,42)`): Deep saturated greens / crimsons for boolean pass/fail badges where the tone needs more weight than the soft semantic family.

#### Score Gradient (Signature)

The product's emotional core. A 0-1 score maps to **`hsl(score × 120, 70%, 45%)`**: 0 → red, 0.5 → yellow, 1 → green. Used as both foreground and translucent background on score chips, dataset chips, and group-header best-score callouts. This is computed inline (`scoreColor` / `scoreBg` helpers), never stored as a static palette. **Treat the formula as a brand asset** — do not reskin to a 5-step bucket, do not introduce a 4th hue.

Foreground uses HSL for predictable hue progression; background uses an RGB-interpolated stop pair (rich mid-tones, more saturated yellows) for visual continuity with the existing chip/badge appearance. Do not rewrite the bg to HSL alpha — the mid-tone shift will break existing visual reads.

**Per-theme legibility knobs**: `hsl(h 70%, 45%)` reads cleanly on the near-black dark canvas but the yellow mid (hue ≈ 60) collapses into warm-cream `#faf9f5` on the light theme. To keep the brand formula intact while fixing legibility, three CSS vars scope per-theme:

- **`--score-fg-s`** — saturation for the HSL foreground. Dark: `70%`. Light: `85%` (the +15pt bump compensates for the lightness compression at hue ≈ 60; without it, mid scores render as washed-out olive on cream).
- **`--score-fg-l`** — lightness for the HSL foreground. Dark: `45%`. Light: `32%` (darker olive/forest/maroon read cleanly on cream).
- **`--score-bg-a-mul`** — alpha multiplier for the RGB-interpolated bg. Dark: `1`. Light: `1.6` (boosts washed-out yellows so the pill bg is actually visible).

`scoreColor` / `scoreBg` emit CSS expressions (`hsl(h var(--score-fg-s) var(--score-fg-l))`, `rgb(r g b / calc(α * var(--score-bg-a-mul)))`) — the browser picks the right value per active theme. **Do not** hardcode these in JS or fork separate light/dark helpers.

**Score Ring** (`{components.score-ring}` — SVG circular progress used in `<ReportSummaryStats>` and the "Overall Score" callout in `<DetailsTab>`): the active arc is `stroke={scoreColor(score)}`. Stroke width must be **≥6 px** for the 48 × 48 mini ring and **8 px** for the 72 × 80 summary ring — anything thinner reduces the colored area to the point where the mid-hue olive stops carrying. Background arc uses `var(--border)` for a neutral track.

#### Compare Slots

Three per-model accent colors used to tag side-by-side model comparisons. Each slot has a `dot`, `border`, `bg`, and `bg-header` tint at ~10-30 % alpha:

- **Slot 0** (`{compare.0.dot}` — `#818cf8` / `#6c57e8`): Indigo.
- **Slot 1** (`{compare.1.dot}` — `#34d399` / `#0a8a6e`): Mint.
- **Slot 2** (`{compare.2.dot}` — `#fbbf24` / `#d97706`): Amber.

If a comparison view exceeds 3 models, *do not invent a 4th brand color* — collapse into a numbered legend instead.

#### Chat Bubble Roles

Five semantic roles, each with a complete 7-token set (`bg`, `bg-hl`, `border`, `border-hl`, `icon-bg`, `icon-border`, `color`):

- **User** — Indigo family (`#818cf8` color).
- **Bot** — Emerald family (`#34d399` color).
- **Tool** — Amber family (`#fbbf24` color).
- **Reasoning** — Dim emerald (lower-saturation companion to Bot).
- **System** — Slate gray (`rgba(148,163,184,*)`).

Bubble containers are `{rounded.md}` with the role's tint background and border; hover/highlight states use the `*-hl` variants.

#### KPI Gradients

Four named linear gradients for the four hero KPI tiles on the dashboard:

- **Indigo→Violet** (`{gradients.kpi-0}` — `linear-gradient(135deg, #6366f1, #8b5cf6)`)
- **Emerald→Teal** (`{gradients.kpi-1}` — `linear-gradient(135deg, #10b981, #06b6d4)`)
- **Amber→Orange** (`{gradients.kpi-2}` — `linear-gradient(135deg, #f59e0b, #f97316)`)
- **Pink→Violet** (`{gradients.kpi-3}` — `linear-gradient(135deg, #ec4899, #8b5cf6)`)

Always applied to the 40 × 40 `{rounded.md}` icon tile inside a `{components.kpi-card}`. **Same gradient values on both themes** — they're saturated enough to work over either canvas.

#### Chart Palette (Perf Metrics)

Four hue tokens used to mark perf-metric series (latency / TTFT / TPOT / token-usage) across the KPI strip, the chart series legends, and the percentile-table accent headers in `<PerfMetricsPanel>`. **The two themes use different RGB values for the same hue** — unlike the KPI gradients, these aren't shared across themes:

- **Latency** (`{chart.latency}` — `#60a5fa` dark / `#2563eb` light)
- **TTFT** (`{chart.ttft}` — `#34d399` dark / `#047857` light)
- **TPOT** (`{chart.tpot}` — `#a78bfa` dark / `#5b48e0` light)
- **Token** (`{chart.token}` — `#94a3b8` dark / `#5a6378` light)

**Why the fork**: warm-cream bg (`{colors.bg-card-light}` / `{colors.bg-deep-light}`) compresses cool hues via simultaneous contrast. Mid-tone slate blue / violet that pop on near-black dark canvases read as washed-out lavender on cream. The light palette pulls each hue darker **and** more saturated to restore the data-bearing punch of the same hue family. Same hue identity, different RGB — not a re-tint.

**KPI strip surface**: the strip in `<PerfMetricsPanel>` uses `{colors.bg-card}` (matching the outer card), not `{colors.bg-deep}`. On light theme, `{colors.bg-deep}` is even warmer than the cards and pushes the chart hues into the warm-on-warm range — losing the contrast that makes the colored numbers carry. Visual separation comes from the border + dividers, not bg differentiation.

#### Brand Gradients (Decorative)

- **Brand** (`{gradients.brand}` — `linear-gradient(135deg, #816DF8 → #a78bfa)`): For `gradient-text` and large brand moments.
- **Accent** (`{gradients.accent}` — `linear-gradient(135deg, #0F9C7E → #06b6d4)`): For the optional emerald-to-cyan accent text.
- **Surface** (`{gradients.surface}` — `linear-gradient(135deg, rgba(129,109,248,0.08) → rgba(167,139,250,0.05))`): The subtle violet wash layered behind KPI cards via `::before`.

### Typography

#### Font Family

Two cross-platform **system font stacks** carry the entire system — each OS resolves to its own native UI face. No `@font-face` is loaded; this is deliberate. Each stack starts with the modern CSS `system-ui` / `ui-monospace` generic family and falls back to named faces for older browsers and per-OS targets:

1. **System sans** (`{typography.font-sans}` — `system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif`): Every display, body, button, link, and label. Working weights: 400 / 500 / 600 / 700. Resolves to **SF Pro** on macOS / iOS, **Segoe UI** on Windows, **Roboto** on Android / ChromeOS, and the desktop default (Cantarell / Noto / DejaVu) on Linux.
2. **System mono** (`{typography.font-mono}` — `ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, "Liberation Mono", "DejaVu Sans Mono", "Courier New", monospace`): Timestamps, scores, model IDs, and any tabular-numeric data. Weight 400 only. Resolves to **SF Mono** on macOS, **Consolas** on Windows, **Liberation Mono / DejaVu Sans Mono** on Linux. The full chain matters — without the OS-specific named fallbacks, non-mac users land on `Courier New`, breaking the tabular-numeric look.

Antialiasing is forced (`-webkit-font-smoothing: antialiased`). No web font is loaded — the brand reads as "native developer tool, not marketing site" precisely because of this. The trade-off is per-OS rendering variance; if pixel-identical screenshots across platforms are required, see *Note on Font Substitutes* below.

#### Hierarchy

| Token | Size | Weight | Tracking | Use |
|---|---|---|---|---|
| `{typography.display-xl}` | 24px | 700 | tight | KPI value, hero number (dashboard `{components.kpi-card}`). |
| `{typography.title-md}` | 16px | 700 | normal | Card-title model name, group-header titles, brand wordmark. |
| `{typography.body-sm}` | 14px | 400 / 500 | normal | Default body text, button-md label, table-cell text, paragraph copy. |
| `{typography.label-xs}` | 12px | 600 | wider | **UPPERCASE** card-section header, form label, badge text — the brand's signature eyebrow. |
| `{typography.body-xs}` | 12px | 400 / 500 | normal | Empty-state hint, pill / badge body, mobile-nav link. |
| `{typography.table-xs}` | 12px | 600 | wider | **UPPERCASE** table-header text; never smaller because headers are essential. |
| `{typography.caption-mono}` | 12px | 400 (mono) | normal | Timestamps, score values, dataset names in chips. |
| `{typography.code}` | 13–14px | 400 (mono) | normal | Log viewer, JSON viewer, terminal-style output. |
| `{typography.button-sm}` | 12px | 500 | normal | `{components.button}` `size="sm"`. |
| `{typography.button-md}` | 14px | 500 | normal | `{components.button}` `size="md"` (default). |
| `{typography.button-lg}` | 16px | 500 | normal | `{components.button}` `size="lg"` — for hero callouts only. |

#### Principles

- **UPPERCASE + `tracking-wider` is the eyebrow voice — never the headline voice.** It marks "this introduces a region." Card titles, model names, and KPI labels stay sentence-case (or lower-case for the brand wordmark's lowercase "v").
- **`tracking-tight` is reserved for the display tier** — KPI numbers and the brand wordmark only. It tells the reader "this is set big on purpose."
- **`tabular-nums` + `font-mono` everywhere numbers must align** — KPI values, score chips, timestamps, percentage cells. Treat numbers as data, not prose.
- **Weight 700 is the display ceiling.** No `font-extrabold` / `font-black`. The brand reads as a calmer system because of this.
- **Line-heights inherit Tailwind defaults** (1.5 for body, 1.25 for headings). Don't override globally; rely on padding for vertical rhythm in tight stacks (cards, chips).
- **No web font.** The system stack is the system. Loading Inter or Geist on top would break the "native console" feel.

#### Note on Font Substitutes

EvalScope uses the OS-native stack, so there are no proprietary faces to substitute. If a future skin needs to enforce a *single* face across all OSes for screenshot consistency:

- **Sans substitute** — *Inter* (400 / 500 / 600 / 700) is the closest stylistic match to the SF-on-macOS rendering; preserves the geometric / neutral character.
- **Mono substitute** — *JetBrains Mono* (400) at 12–14 px matches the technical voice well; *IBM Plex Mono* is a close second.

### Layout

#### Spacing System

- **Base unit**: 4 px (Tailwind's default scale).
- **Tokens** (Tailwind-aligned):
  - `{spacing.xs}` 4 px · `{spacing.sm}` 8 px · `{spacing.md}` 12 px · `{spacing.lg}` 16 px · `{spacing.xl}` 20 px · `{spacing.2xl}` 24 px · `{spacing.3xl}` 32 px · `{spacing.4xl}` 48 px · `{spacing.5xl}` 64 px.
- **Page padding**: 16 px horizontal, 20 px vertical (`px-4 py-5`) at all breakpoints — *no responsive expansion*. The 1600 px container does the breathing for us.
- **Card interior padding**: 20 px body (`{spacing.xl}`), 12 px header strips (`{spacing.md} {spacing.xl}`).
- **Inline gap**: `{spacing.md}` (12 px) for component rows inside a card, `{spacing.xl}` (20 px) for inter-section gaps between major page blocks.
- **Pill / chip gap**: `{spacing.sm}` (6-8 px) — tight, scan-friendly. Pills are meant to wrap.

#### Grid & Container

- **Max width**: `1600px` (`max-w-[1600px]`). Centered (`mx-auto`). Wide enough for 4-up KPI strip + side-by-side comparison; narrow enough that line-length never sprawls.
- **Column patterns**:
  - **KPI strip**: 4-up at `lg+`, 2-up below (`grid-cols-2 lg:grid-cols-4`).
  - **Compare view**: 2-3 columns at desktop, vertical stack on mobile.
  - **Eval timeline**: single column of full-width rows, no grid.
  - **Form pairs**: `grid-cols-1 md:grid-cols-2` for label-value pairs.
- **Gutters**: 16 px horizontal at all sizes.

#### Whitespace Philosophy

Whitespace separates the *bands* — not the components inside a band. Section spacing is generous (`flex flex-col gap-5` → 20 px between major blocks); card interiors are tight (`gap-2` / `gap-3` between rows inside a card). The page reads as engineered — *large gaps + tight interior, never the other way around*. The dark page lets cards float visually without needing margin to assert themselves; the hairline border does the bordering work.

The brand's voice is **information-rich but uncluttered** — a typical dashboard packs a 4-tile KPI strip, a 30-row sortable eval list, search + sort + view-toggle controls, and the page still feels scannable because:
1. Tokens enforce a 3-step text hierarchy on every row.
2. UPPERCASE eyebrows visually section the layout without horizontal rules.
3. Score chips compress a percentage + benchmark name + color signal into 6 characters of mono.

#### Responsive Strategy

##### Breakpoints (Tailwind defaults)

| Name | Width | Key Changes |
|---|---|---|
| Mobile | < 640px | Nav collapses to hamburger drawer; KPI grid drops to 2-up; all controls stack. |
| Small | 640–767px | GitHub icon and locale toggle appear in nav; KPI still 2-up. |
| Tablet | 768–1023px | Nav switches to **icon-only mode** with tooltips; main content full-width. |
| Desktop | 1024–1279px | Full pill-style nav with icon + label; KPI strip goes 4-up. |
| Wide | ≥ 1280px | Container holds at `max-w-[1600px]`; bands stretch but content centers. |

##### Touch Targets

**Normative rule (executable, enforced — see [Component Contracts](#component-contracts)).** Every primary `Pointer_Target` — navigation, mobile menu, compare-selection, and disclosure controls — MUST expose a hit area of **≥ 44 × 44 CSS px on coarse pointers** (Req 12.5, 12.7). This is a hard floor asserted by the E2E/axe suite at the 390 px viewport, **not** a documented known-failure or accepted compromise.

The *visual* chrome may be smaller than the *hit area*, and only the hit area is governed by the rule. The top-nav icon-only buttons (tablet) render a 32 × 32 visual box, but on coarse pointers their tappable region is expanded to ≥ 44 × 44 via symmetric padding / an `::before` hit-area overlay — the icon stays 32 × 32, the target does not shrink below the floor. Primary buttons reach ~36 px tall in `md` and ~44 px in `lg`; where an `md` button is a primary `Pointer_Target` on touch, it is promoted to the 44 px hit area by the same rule.

##### Collapsing Strategy

- **Nav**: Desktop = pill-row with icon + label; tablet = icon-only pills; mobile = logo + hamburger toggling a stacked drop-down (`max-height` animated, 300 ms).
- **KPI strip**: 4-up → 2-up below `lg`. Each tile keeps its `{rounded.md}` 12-px shape and 20-px padding.
- **Eval timeline**: Single column at all sizes — already a vertical list.
- **Forms**: Two-column label/value at `md+`, single column below.
- **Table**: Horizontal scroll wrapper preserves all columns rather than dropping them. The card-shell border keeps the scroll area framed.

##### Image / Icon Behavior

- **Iconography**: `lucide-react`, almost always 14–18 px, color inherits `currentColor`. The only non-Lucide mark is the brand SVG in the top-nav (a hand-drawn triangle with an amber check, rendered with `currentColor` so it follows theme).
- **No marketing imagery**: This is a console — no hero photo, no customer-logo strip, no illustrated empty-states. Empty states use a single Lucide icon in a 64 × 64 `{rounded.lg}` deep-well tile.
- **Charts**: Plotly-based, theme-aware — chart series colors are `{colors.chart-*}` tokens (latency / TTFT / TPOT / token).

### Elevation & Depth

| Level | Treatment | Use |
|---|---|---|
| **L0 — Flat** | No border, no shadow. | Page body, large empty regions, full-bleed dark sections. |
| **L1 — Hairline** | 1-px `{colors.border}` only. Dark: **translucent violet at 10 % alpha** (the near-black page lets even a faint hairline read; the slight violet tint stays on-brand). Light: **solid warm-grey hex** `#e6dfd8` (translucent violet at any plausible alpha composites to invisible against a white card on cream — solid hex is the only thing that reads). | Default content cards (`{components.card}`), table chrome, form inputs. |
| **L2 — Lifted** | `{shadows.sm}` + L1 hairline. Dark: single deep drop `0 2px 8px rgba(0,0,0,0.4)`. Light: two-stop stack `0 1px 2px rgba(20,20,19,0.04), 0 4px 12px rgba(20,20,19,0.06)` — warm-ink tinted (matches `{colors.text-light}`), not slate. Slate-tinted drops on cream read as a cool-grey smudge against the warm canvas. | KPI cards, path-bar at the top of the dashboard, anything that needs to read as "above the page." |
| **L3 — Floating** | `{shadows.md}` + L1 hairline. Dark `0 4px 20px rgba(0,0,0,0.55)`. Light `0 4px 16px rgba(20,20,19,0.07), 0 12px 32px rgba(20,20,19,0.05)`. | Default for the primary card variants when the page already has heavy chrome. |
| **L4 — Elevated** | `{shadows.lg}` + `{colors.border-strong}`. Dark `0 8px 40px rgba(0,0,0,0.6)`. Light `0 12px 24px rgba(20,20,19,0.09), 0 24px 48px rgba(20,20,19,0.07)`. | Hover state for clickable cards, modal / dialog surfaces, dropdown menus. |
| **L5 — Glow** | `{shadows.glow}` on top of L1-L3. Dark `0 0 20px rgba(129,109,248,0.25)`. Light `0 0 20px rgba(108,87,232,0.22)`. | The signature violet halo — primary button hover, active nav pill, sometimes active tab. Identical magnitude across themes — the glow is the brand's interaction constant. |

**Brand rule (depth)**: dark and light themes are *not* the same shadow scaled down. Dark surfaces use a **single deep drop** (`rgba(0,0,0,0.4-0.6)`) — the near-black canvas eats soft drops. Light surfaces use a **two-stop stack** tinted with warm-ink `rgba(20,20,19,*)` (the same hex as `{colors.text-light}`), NOT slate `rgba(15,23,42,*)` — the warm-ink tint stays coherent with the cream canvas and the warm-ink body text. Light cards lift through the *combination* of cream-to-white surface contrast + a solid warm-grey hairline + the two-stop warm-ink shadow stack — no single layer carries the weight.

**Brand rule (hairlines)**: light theme borders are **SOLID HEX**, not translucent violet at any alpha. The earlier light-theme generation used `rgba(108,87,232,0.20-0.40)` and the violet alpha composited to within a few luminance steps of the white card — borders effectively dissolved, especially on outline buttons (`Go to Index`, `Find msg id`) and on input rings (`Score Threshold` field). The current system uses three concrete warm-grey hex values (`#e6dfd8` / `#d6cdbe` / `#c1b6a3`) — each step is a definite material, not a tint. The dark theme keeps translucent violet at 10-28 % because the indigo-bg-to-card luminance delta is already doing most of the boundary work; light theme has no such luminance assist and must rely on the hairline alone.

#### Decorative Depth

- **Backdrop-blur**: 12 px on `{colors.surface-glass}` for the sticky top-nav. This is the only blur effect in the system.
- **Hairline gradient line**: The top of the nav draws a 1-px `transparent → {colors.accent} → transparent` line at 40 % opacity. The closest the design comes to "decoration."
- **Card lift**: Hover lifts L1/L2 cards `-2px` (`{components.card-hover}`) or `-3px` (`{components.kpi-card}`), simultaneously upgrading their shadow ladder one step.
- **Gradient text**: `.gradient-text` and `.gradient-text-accent` utilities apply `{gradients.brand}` / `{gradients.accent}` to text via `background-clip: text`. Use sparingly — reserved for hero brand moments, never for body or table cells.

### Shapes

#### Border Radius Scale

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

#### "Photography" — Iconography Geometry

- **Brand mark**: Hand-drawn SVG triangle with an amber check; rendered inline at 28 × 25 px with `currentColor` so it follows theme.
- **Lucide icons**: 14 px in dense lists, 16 px in nav, 18 px in form controls, 28 px in empty-state hero tiles. Stroke width 1.5–2.
- **KPI icon tile**: 40 × 40, `{rounded.md}`, filled with one of `{gradients.kpi-0..3}`, white icon centered.
- **Chart**: Plotly canvas, no rounded corners; sits inside a `{rounded.md}` card frame.

## Components {#components}

> **Addressable section — `components`.** The component contracts built from the tokens above: buttons, cards, inputs, tabs, navigation, tables, badges, and the signature composite surfaces. Each entry references tokens by name so a re-skin propagates automatically.

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
- **Header cells**: `{typography.table-xs}` — 12 px, semibold, UPPERCASE, `tracking-wider`, `{colors.text-muted}`. Headers are essential interpretation text and therefore meet the 12 px/AA floor. Sortable headers show a triple-state chevron (`ChevronsUpDown` idle, `ChevronUp/Down` active) and lift to `{colors.text}` on hover; active sort column turns `{colors.accent}`.
- **Row dividers**: 1-px `{colors.border}`. Clickable rows hover-fill `{colors.bg-card2}`.
- **Empty state**: Centered, dimmed "No data" cell — no illustration.

### Badges, Chips & Pills

**`{components.badge}`** — fully rounded inline pill.
- `{rounded.full}`, 8/2 padding, `{typography.body-xs}`. Four variants pair an 8-10 % alpha background with a saturated foreground: `default → accent`, `success → green`, `warning → yellow`, `danger → red`.

**`{components.filter-chip}`** — dismissable filter chip.
- Same pill shape as `{components.badge}`, with an optional 3.5 × 3.5 circular dismiss button (`X` icon).

**`{components.score-chip}`** — the signature dynamic-score pill.
- `{rounded.full}`, 8/2 padding, `{typography.caption-mono}`, **outline treatment**: transparent bg + 1-px `scoreColor` border + `scoreColor` text. Format: `"benchmark-name 87.3"` — the chip *is* the data point. Used in the eval timeline, dashboard tiles, and the leaderboard rows.
- Outline (not filled) on purpose: a filled pill at hue ≈ 60 paints high-luminance yellow (`rgb(255, 255, 0)`) which dominates whatever surface it sits on, regardless of theme. Outline keeps hue legible while letting the chip recede into the row.

### Signature Components

**`{components.kpi-card}`** — see *Cards & Containers* above.

**`{components.eval-run-card}`** — a full-width borderless button-styled card row.
- Two rows of content: bold model name + colored score pill on the right; second row has a mono timestamp + a wrap of `{components.score-chip}` entries (one per benchmark). Hover only changes border tint (no lift).

**`{components.chat-bubble}`** — five-role chat surface.
- `{rounded.md}` container with role-specific bg / border / icon-bg / icon-border / color from the `{colors.bubble-*}` token family (user / bot / tool / reasoning / system). Hover strengthens border via `*-hl` variants.

**Two variants**:
- `bar` (default in chat lists) — `rounded-sm` container, 3-px left vertical accent in the role's color, role's tint as bg. Used by the streaming chat log where many bubbles stack and visual weight must stay low.
- `card` — `rounded-md` container with role's full bg + 1-px border. Used in standalone bubbles outside a scrolling list (eval result preview, single-message dialogs).

**`{components.empty-state}`** — first-contact / no-data state.
- Vertical-center stack: 64 × 64 `{rounded.lg}` deep-well tile holding a 28-px Lucide icon (`{colors.accent}` for welcome states, `{colors.text-dim}` for empty/no-results), followed by a 2-line message — title in `{typography.body-sm}` `{colors.text}` (welcome) or `{colors.text-muted}` (empty), hint in `{typography.body-xs}` `{colors.text-dim}`.

**`{components.path-bar}`** — the dashboard's "scan this directory" input row.
- `{components.card}` chrome at L2, hosting an icon, an input, and a primary button — flex-row with 12-px gap.

**`{components.score-badge}`** — the bold percentage pill at the top of an eval row.
- `{rounded.full}`, 10/2 padding, `{typography.body-sm}` bold + `tabular-nums`, HSL-computed fg/bg. Distinct from `{components.score-chip}` by size and weight.

### Component Contracts (Responsive & State) {#component-contracts}

> **Addressable sub-section — `component-contracts`.** Every responsive component below carries two contracts, and both are **normative and executable** (Req 7.3, 12.7, 17.6) — not descriptive prose:
>
> - a **Responsive contract** — the layout, wrapping, and visibility behavior at each of the five breakpoints (**Mobile < 640 px · Small 640–767 px · Tablet 768–1023 px · Desktop 1024–1279 px · Wide ≥ 1280 px**, per [Responsive Strategy](#tokens)'s breakpoint table), and
> - a **State contract** — the presentation requirement for every visible state (**default · hover · focus · active · disabled · error**). A state a component cannot enter is listed as *n/a* with the reason, so the enumeration stays complete.

#### Normative contract rules (executable, not aspirational)

These thresholds are **enforceable rules, not logged known-failures** (Req 12.7). The E2E/axe suite (task 19.2) asserts them at the 390 px viewport and in both themes; a violation fails CI. Each component contract below inherits them.

- **R-TOUCH — Touch-target floor 44 × 44 CSS px.** On coarse pointers, every primary `Pointer_Target` in navigation, the mobile menu, compare-selection, and disclosure controls MUST expose a hit area **≥ 44 × 44 CSS px**, independent of the visual icon/box size (padding or an `::before` overlay carries the extra area). (Req 7.4, 12.5)
- **R-CONTRAST — AA contrast floor.** Essential text MUST meet **AA_Contrast in both themes**: **≥ 4.5 : 1** for normal text and **≥ 3 : 1** for large text (≥ 24 px, or ≥ 18.66 px bold). `dim`/`muted` tokens applied to essential content are promoted to clear this floor; non-text focus indicators MUST reach **≥ 3 : 1** against the adjacent background. (Req 12.1–12.6)
- **R-WRAP — No lossy truncation of essential metadata.** Responsive text containers use `break-words` + `min-w-0` and MUST NOT apply `truncate`/ellipsis/overflow-hidden to essential metadata; at 390 px every field label and value stays visible and wraps at word (then character) boundaries with no page-level horizontal scroll. (Req 7.1, 7.2)
- **R-FOCUS — Visible focus in both themes.** Every interactive component's **focus** state renders a visible focus indicator (`{colors.accent}` ring + `{colors.accent-dim}` fill) meeting R-CONTRAST's ≥ 3 : 1 non-text floor on both themes. Individual state contracts reference this rule rather than restating it.

#### `{components.top-nav}` / `{components.nav-link}` / `{components.icon-button}` — navigation

- **Responsive contract:**
  - **Mobile < 640:** logo + hamburger only; nav items collapse into a stacked drop-down (animated `max-height`, 300 ms); GitHub link + locale toggle live inside the drawer. Drawer toggle and every drawer item satisfy R-TOUCH.
  - **Small 640–767:** hamburger drawer retained; GitHub icon and locale toggle surface directly in the bar.
  - **Tablet 768–1023:** icon-only pill buttons (32 × 32 visual) with `title` tooltips; text labels hidden; hit area ≥ 44 × 44 per R-TOUCH.
  - **Desktop 1024–1279:** full pill links with icon + label; active = solid `{colors.accent}` + glow.
  - **Wide ≥ 1280:** identical to Desktop; container caps at `max-w-[1600px]`, bar content centers.
- **State contract:**
  - **default:** `{colors.text-muted}` label on transparent fill.
  - **hover:** label lifts to `{colors.text}`, fill → `{colors.bg-card2}`.
  - **focus:** per R-FOCUS.
  - **active (current route):** `{colors.accent}` fill, `{colors.on-filled}` label, `{shadows.glow}` halo.
  - **disabled:** *n/a* — nav destinations are always actionable.
  - **error:** *n/a* — navigation carries no validation state.

#### `{components.button}` — buttons (primary / ghost / outline)

- **Responsive contract:** intrinsic width, wraps within its container per R-WRAP; size is chosen by usage, not breakpoint (`sm` ~28 px, `md` ~36 px, `lg` ~44 px). A primary `Pointer_Target` button on coarse pointers is promoted to the 44 px hit area (R-TOUCH) regardless of visual size. Toolbar button rows wrap to a new line below `md` rather than horizontally scrolling.
- **State contract:**
  - **default:** variant fill/border per *Buttons* above; label meets R-CONTRAST (`{colors.on-filled}` on primary, `{colors.text}` on ghost/outline).
  - **hover:** primary adds `{shadows.glow}` + `bg → {colors.accent-dark}`; ghost → `{colors.bg-card}`; outline swaps border + text to `{colors.accent}`.
  - **focus:** per R-FOCUS.
  - **active (press):** scales to 0.98; ghost → `{colors.bg-card2}`.
  - **disabled:** `opacity: 0.5` + `cursor: not-allowed`; no hover/glow; still announced to AT.
  - **error:** *n/a* — buttons carry no intrinsic error state (submit errors surface on fields).

#### `{components.card}` / `{components.card-hover}` / `{components.row-card}` — cards & rows

- **Responsive contract:** full-width within its band at every breakpoint; interior text follows R-WRAP (no truncation). Card interior padding is fixed 20 px at all breakpoints (page breathing comes from the 1600 px container, not padding expansion).
- **State contract:**
  - **default:** `{colors.bg-card}`, 1-px `{colors.border}`, `{shadows.sm}` (L2).
  - **hover (clickable only):** `{components.card-hover}` lifts −2 px, shadow → `{shadows.lg}` (L4), border → `{colors.border-strong}`; `{components.row-card}` only tints border → `{colors.border-md}` (no transform, kept calm for dense lists).
  - **focus (clickable only):** per R-FOCUS.
  - **active (clickable only):** border holds at `{colors.border-strong}`; no additional transform.
  - **disabled:** *n/a* — non-interactive cards have no disabled state; a disabled clickable card falls back to the button disabled contract.
  - **error:** *n/a* — error is expressed by the content inside (e.g. `{components.empty-state}` load-error), not the card shell.

#### KPI strip / `{components.kpi-card}` — hero metric tiles

- **Responsive contract:**
  - **Mobile < 640 & Small 640–767:** 2-up grid (`grid-cols-2`).
  - **Tablet 768–1023:** 2-up grid retained; tiles full-width of their column.
  - **Desktop 1024–1279 & Wide ≥ 1280:** 4-up grid (`lg:grid-cols-4`). Each tile keeps `{rounded.md}` and 20-px padding at every breakpoint; the value uses `tabular-nums` and wraps per R-WRAP rather than truncating.
- **State contract:**
  - **default:** `{components.card}` chrome + `{gradients.surface}` wash; 40 × 40 gradient icon tile.
  - **hover:** lifts −3 px, ramps to L4 shadow.
  - **focus:** per R-FOCUS when the tile is interactive; *n/a* for display-only tiles.
  - **active:** *n/a* unless linked, then follows the clickable-card active contract.
  - **disabled:** *n/a*.
  - **error:** *n/a* — a metric with no value renders the missing-value placeholder inside the tile (see [Metric Display Contract](#metrics)), not a tile error state.

#### `{components.tabs}` — pill-container tabs

- **Responsive contract:** segmented pill row at `md+`; below `md` the row wraps (or scrolls its own container, never the page) per R-WRAP so every tab label stays legible. Tab labels never truncate.
- **State contract:**
  - **default (inactive):** `{colors.bg-card}` fill, `{colors.text-muted}` label; `tabindex=-1` under roving-tabindex.
  - **hover:** fill → `{colors.bg-card2}`.
  - **focus:** the single active tab holds `tabindex=0`; arrow keys move focus with wrap-around; per R-FOCUS.
  - **active (selected):** `{colors.accent}` fill + `{colors.on-filled}` text + soft glow `0 0 12 px rgba(129,109,248,0.2)`; exactly one tab has `aria-selected=true` and one visible `tabpanel`.
  - **disabled:** disabled tab renders at `opacity: 0.5`, `aria-disabled=true`, skipped by roving navigation.
  - **error (orphan item):** a tab with no matching `tabpanel` reference is **not rendered** and an error is surfaced (Req 11.8).

#### `{components.table}` — sortable data table

- **Responsive contract:**
  - **Mobile < 640 & Small 640–767:** the Evaluation-History surface presents the **card layout** (field-per-line, R-WRAP, no truncation); a raw table, where shown, is wrapped in `overflow-x-auto` so all columns are preserved (horizontal scroll inside the card shell, never at the page level).
  - **Tablet 768–1023:** table with `overflow-x-auto`; all columns retained rather than dropped.
  - **Desktop 1024–1279 & Wide ≥ 1280:** full table with the fixed, ordered columns `model · dataset · time · samples · score · status`. Header text uses `{typography.table-xs}` at 12 px and is never shrunk below the floor.
- **State contract:**
  - **default:** header `{typography.table-xs}` UPPERCASE `{colors.text-muted}`; 1-px `{colors.border}` row dividers.
  - **hover:** sortable header lifts to `{colors.text}`; clickable row fills `{colors.bg-card2}`.
  - **focus:** sortable header / row focusable, per R-FOCUS.
  - **active (sorted column):** header turns `{colors.accent}` with a `ChevronUp/Down` direction indicator.
  - **disabled:** *n/a* — columns are always sortable/among the fixed set.
  - **error / empty:** centered dimmed "No data" cell (or the `{components.empty-state}` surface when the whole view is empty).

#### `{components.input}` / `{components.select}` / `Field_Primitive` — inputs & forms

- **Responsive contract:** label/value pairs are two-column at `md+` (`grid-cols-1 md:grid-cols-2`) and single-column below `md`; label and helper text follow R-WRAP. Inputs are full-width of their column at every breakpoint.
- **State contract:**
  - **default:** `{colors.bg-deep}` well, 1-px `{colors.border}`, `{colors.text}` value, `{colors.text-dim}` placeholder; a non-empty programmatically-associated label (`{colors.text-muted}`, UPPERCASE `{typography.label-xs}`).
  - **hover:** border → `{colors.border-md}`.
  - **focus:** border → `{colors.accent}` + 1-px `{colors.accent-dim}` ring (soft halo); per R-FOCUS.
  - **active:** same as focus while editing.
  - **disabled:** `opacity: 0.5`, `cursor: not-allowed`, `aria-disabled`; still exposes its accessible name.
  - **error:** border + ring swap to the danger family, `aria-invalid=true`, `aria-describedby` points at a 12 px `{colors.danger}` helper line; the message is announced via a polite live region within 1 s and receives focus first among invalid fields on submit (Req 10.3, 10.4).

#### `{components.empty-state}` — actionable empty states

- **Responsive contract:** vertically-centered stack, full-width of its band at every breakpoint; the 64 × 64 icon tile and the two message lines wrap per R-WRAP and never truncate. Recovery-action buttons stack below `md` and sit inline at `md+`.
- **State contract:**
  - **default:** icon tile (`{colors.accent}` welcome / `{colors.text-dim}` empty) + title + hint; renders within 300 ms of a completed zero-record load, with reason-specific text for `no-data` / `load-error` / `no-match`.
  - **hover / focus / active:** carried by the 1–3 embedded recovery-action buttons per the `{components.button}` contract (each a `Pointer_Target` ≥ 44 × 44 on touch, R-TOUCH); every action has a non-empty `navigateTo`.
  - **disabled:** *n/a* — recovery actions are always actionable.
  - **error:** the `load-error` reason **is** this component's error presentation (distinct copy + retry action); it does not render a blank region.

#### `{components.eval-run-card}` — eval timeline run row

- **Responsive contract:** single full-width column at every breakpoint (the timeline is already a vertical list). At 390 px all metadata fields (model, dataset, time, samples, score, status, per-benchmark score chips) stay visible and wrap per R-WRAP — no ellipsis, no horizontal scroll (Req 7.1, 7.2). The score-chip row wraps onto multiple lines rather than clipping.
- **State contract:**
  - **default:** `{colors.bg-card}`, 1-px `{colors.border}`; bold model name + score badge; mono timestamp + wrapped `{components.score-chip}` set.
  - **hover:** border tint → `{colors.border-md}` only (no lift, calm for dense lists).
  - **focus:** per R-FOCUS (the row is a button).
  - **active (press):** border holds at `{colors.border-md}`; navigates to detail.
  - **disabled:** *n/a*.
  - **error:** *n/a* — a run with missing values shows placeholders/`N/A` per the metric contract, not a card error state.

#### `{components.score-chip}` / `{components.badge}` / `{components.filter-chip}` — chips & pills

- **Responsive contract:** chips are meant to **wrap** — a chip row flows onto multiple lines with an 8-px gap at every breakpoint and never truncates or horizontally scrolls (R-WRAP). Chip content (`"benchmark-name 87.3"`) is `tabular-nums` mono and stays whole.
- **State contract:**
  - **default:** `{components.score-chip}` outline (transparent bg + 1-px `scoreColor` border + `scoreColor` text); `{components.badge}` uses an 8–10 % alpha bg + saturated fg. Both meet R-CONTRAST for the essential value.
  - **hover:** `{components.filter-chip}` reveals its dismiss control; score/badge pills are static.
  - **focus:** `{components.filter-chip}` dismiss control and any interactive chip follow R-FOCUS and R-TOUCH (≥ 44 × 44 hit area on touch).
  - **active:** dismiss removes the chip; static chips have no active state.
  - **disabled:** *n/a*.
  - **error:** *n/a*.

#### `ex-compare-column` / `{components.card}` (compare) — compare-view model column

- **Responsive contract:**
  - **Mobile < 640 & Small 640–767:** columns stack vertically (one model per full-width block); display label and metric rows wrap per R-WRAP.
  - **Tablet 768–1023:** 2 columns side-by-side.
  - **Desktop 1024–1279 & Wide ≥ 1280:** 2–3 columns side-by-side; beyond 3 models the view collapses into a numbered legend rather than inventing a 4th slot color.
- **State contract:**
  - **default:** header carries the model+dataset display label and one of the `{compare.0..2}` slot accents (`dot` / `border` / `bg-header`).
  - **hover:** column header/row tint strengthens via the slot's `bg-hl` where interactive.
  - **focus:** compare-selection controls follow R-FOCUS and R-TOUCH.
  - **active (selected for compare):** slot accent applied; selection persists across sort/filter and is reflected in the sticky selection tray count.
  - **disabled (selection limit / incompatible):** at `MAX_COMPARE_SELECTION = 5` further checkboxes are disabled with an explanatory message; an incompatible run shows its incompatibility reason and retains existing selection (Req 5.9, 5.10).
  - **error:** incompatibility reason text is the error presentation; it does not drop the selection.

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

## Metric Display Contract {#metrics}

> **Addressable section — `metrics`.** How raw metric values become on-screen text. Display form is decided by a metric's `MetricDisplaySpec` metadata — never by inspecting the numeric magnitude — so the same value renders consistently across list, detail, compare, and export surfaces. The contract below is implemented by `src/domain/metric/MetricDisplaySpec.ts` and `src/domain/metric/metricFormat.ts`.

### Display Rules

- **Spec-driven, not value-driven.** `resolveSpec(key, registry)` returns the metric's spec (or `DEFAULT_METRIC_SPEC` with `isFallback = true`). The spec's `kind` — not the value — selects the presentation. A raw value greater than `1` never triggers percentage formatting.
- **Bounded ratio → percentage.** A `bounded-ratio` metric (domain `0–1`) renders as a percentage to `percentPrecision` decimals using round-half-up, and preserves a 4-decimal raw value alongside it. `bounded` metrics (domain `0–100`) display on their native scale.
- **Unbounded / native → raw with unit.** Unbounded or native-unit metrics keep their unit and render at `rawPrecision`; they are never converted to a percentage.
- **Missing vs. legitimate zero.** A missing value renders `MISSING_PLACEHOLDER` and sets `isMissing = true`; a real `0` renders as a formatted zero. The two are always distinguishable.
- **Undefined spec fallback.** When no spec is registered for a key, the value renders as a 4-decimal raw number with `isSpecUndefined = true`, and the implementation-level metric name is hidden behind a localized `labelKey`.
- **Rounding.** All decimal rounding uses `roundHalfUp(value, precision)` so `.5` cases round up deterministically across every surface.

### Threshold Semantics

- **Prediction threshold is a view-only filter.** A per-view threshold only annotates or filters rows (`Below filter` / `Above filter`); it never participates in pass/fail and never affects other views.
- **Native outcome is independent.** A benchmark's native `outcome` (pass/fail) is derived and displayed independently of any prediction threshold.

## Decision Records {#decisions}

> **Addressable section — `decisions`.** The load-bearing decisions behind this system, captured so future changes understand what is intentional versus incidental. Each decision is normative; the *Do's and Don'ts* that follow are the enforceable expression of these decisions.

### D1 — Dual-theme parity, not translation

Dark and light are two equally weighted voices of one brand, not one canonical mode re-tinted. They share vocabulary (type, spacing, radii, components) but use *opposite materials* to produce the same hierarchy: dark uses translucent-violet hairlines on near-black; light uses solid warm-grey hairlines on cream. Every new color token must define both a dark and a light value in the same commit.

### D2 — Solid-hex hairlines on the light theme

Light-theme borders are solid warm-grey hex (`#e6dfd8` / `#d6cdbe` / `#c1b6a3`), never translucent violet. Violet at any plausible alpha composites to near-invisible against a white card on cream. This is the single most-broken light-theme rule; see [Design Tokens → Hairlines](#tokens).

### D3 — Dynamic HSL score gradient is a brand asset

Scores map to `hsl(score × 120, 70%, 45%)` (red → yellow → green), computed inline and never stored as a static palette. Per-theme legibility is tuned only through CSS vars (`--score-fg-s`, `--score-fg-l`, `--score-bg-a-mul`); the formula itself is not re-skinned into buckets and gains no fourth hue.

### D4 — System font stack, no web font

The brand loads no `@font-face`; cross-platform `system-ui` / `ui-monospace` stacks resolve to each OS's native face. This is deliberate — it reads as "native developer tool," and the trade-off is per-OS rendering variance.

### D5 — Token single source of truth and drift governance

Token values live once in `evalscope/web/src/index.css`. Run `npm run design:tokens` after changing shared runtime tokens; `scripts/drift/tokenDrift.ts` verifies that this document's frontmatter was generated from the current CSS values.

### D6 — Additive, reversible migration

Refactors preserve raw values, keep chart data-table fallbacks, retain old components until visual-regression parity is proven, and let `zod` schemas coexist with hand-written interfaces during transition.

### Do's and Don'ts

#### Do

- Reserve `{colors.accent}` (`#816DF8`) for primary CTAs, active states, focus rings, and the wordmark accent. Brand violet IS the conversion target — keep it under ~10 % of any screen.
- Use `{rounded.sm}` 8 px for buttons / inputs / tabs, `{rounded.md}` 12 px for cards, `{rounded.full}` only for badges / chips / score pills. Each shape signals its category.
- Set every section eyebrow, form label, and table header in `{typography.label-xs}` — UPPERCASE + `tracking-wider`. This is the brand's hierarchy signal; without it, the design flattens.
- Use `{typography.caption-mono}` + `tabular-nums` for any numeric column — scores, timestamps, percentages. Treat numbers as data, not prose.
- Compute score colors with `hsl(score × 120, 70%, 45%)` and use the same formula for foreground and a translucent background. The dynamic chip is the product's emotional signal.
- Layer stacked shadows (a deep multi-stop shadow + a 1-px hairline) rather than single heavy drops. Cards sit on the page, not above it. On dark the hairline is translucent violet; on light the hairline is solid warm-grey hex — see `{colors.border}`.
- Cycle page chrome through the surface ladder `{colors.bg}` → `{colors.bg-deep}` → `{colors.bg-card}` → `{colors.bg-card2}`. The ladder semantics are theme-agnostic: inputs sit *deeper* than cards, hover sits *higher* — even though dark walks an indigo ladder and light walks a warm-cream ladder.
- **Theme parity:** when adding a new colour token, define BOTH a dark and a light value at the same time, in the same commit. The light value is not a translation of the dark — pick it for the warm-cream context. Token names without a light pair will eventually fall back to a default that breaks one of the themes.
- Animate page transitions with `fadeInUp` (12 px translate + opacity, 400 ms ease-out) and stagger children at 60 ms. The motion is subtle — don't lengthen it.
- Persist the `data-theme` to `localStorage` and apply it pre-paint in `index.html` to avoid FOUC. Tokens are theme-agnostic by name; values flip.

#### Don't

- Don't introduce a 6th brand hue or a 4th compare-slot color. The palette is closed at violet + emerald + amber + red + slate (plus the dynamic HSL score). New accents flatten the voice.
- Don't render headlines in all-caps. UPPERCASE is the 12 px eyebrow/table-label voice — never the title voice. Card titles and model names stay sentence-case.
- Don't promote the sans to `font-extrabold` / `font-black`. The display weight ceiling is **700**.
- Don't use `{colors.text-dim}` for essential UI text on either theme — its ~3.6 : 1 contrast against `{colors.bg-card}` clears AA Large (3 : 1) but is below WCAG AA Normal (4.5 : 1). Reserve textual use for ≥ 14 px non-essential metadata (timestamps and empty labels). Placeholder text, decorative icons, separators, disabled controls, and scrollbar thumbs are exempt because they are not essential readable content. Add the inline note `// text-dim allowed: non-essential ≥14px metadata (DESIGN.md §Text)` when a textual use is not self-evidently a placeholder or empty-state annotation; do not scatter the note onto every decorative icon.
- Don't reuse a dark-theme shadow value verbatim on light. The light palette stacks two **warm-ink-tinted** drops (`rgba(20,20,19,0.04)` + `rgba(20,20,19,0.06)`) — a single `rgba(0,0,0,0.07)` drop on cream reads as a page smudge, not as a lifted card. Do not slate-tint the light shadows either (`rgba(15,23,42,*)`) — slate on cream reads as a cool-grey smear that fights the warm canvas. See *Elevation & Depth*.
- **Don't use translucent violet for light-theme hairlines.** This is the most-broken light-theme rule. The light `{colors.border-light}` is a SOLID warm-grey hex (`#e6dfd8`) — translucent violet at *any* plausible alpha (0.10 / 0.20 / 0.30 / 0.40) composites to near-invisible against a white card on cream and leaves every card boundary undefined. Outline buttons and input rings will vanish. The dark theme uses translucent violet because the near-black bg-to-card luminance delta carries the boundary; the light theme has no such delta and must use a concrete material.
- Don't introduce a cool-grey or pure-white surface to the light theme. The light palette is warm-cream by design (`#faf9f5` canvas, `#f0ebe1` deep, `#f5f0e7` elevated, `#ffffff` cards). A cool-grey `#f5f6fa` or `#eef0f7` band breaks the warm-coherent rhythm and reverts the system to "any other AI dashboard."
- Don't drop a single heavy 8-px-blur drop-shadow on a card. The dark theme requires *deeper* multi-stop shadows (`rgba(0,0,0,0.55)` at 20-40 px) — soft drops disappear on near-black.
- Don't apply `{gradients.brand}` to body text or table cells. Gradient-text is for hero / wordmark moments only.
- Standalone actions and conversion CTAs use `{components.button}` variants; do not re-derive their glow, press, focus, or disabled behavior with a one-off `bg-[var(--accent)]`. Component-internal controls with a distinct semantic contract (tabs, segmented selectors, list rows, score chips) may compose token classes directly, but their states belong in that component's contract and tests.
- Prefer token-backed Tailwind classes or `formStyles` for static surfaces. Inline styles are reserved for values that are genuinely computed at runtime (for example score colors, chart series colors, progress widths, and measured geometry); they must still resolve through documented tokens/formulas where applicable.
- Don't pair `{rounded.full}` pill shapes with `{rounded.md}` cards as siblings inside the same control group — pills are for *data*, sm/md radii are for *interactive containers*. Mixing them on the same row breaks the shape grammar.
- Don't ignore `prefers-color-scheme` on first visit. If the user has never toggled, fall back to the OS preference before defaulting to dark.
- Don't widen `{spacing.xl}` (20 px) section gaps past `{spacing.2xl}` (24 px). The product is information-dense by design; extra whitespace makes the dashboard feel half-empty rather than airy.

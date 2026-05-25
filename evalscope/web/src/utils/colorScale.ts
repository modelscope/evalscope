/**
 * Score → color helpers.
 * The product's emotional core — see DESIGN.md §Score Gradient.
 *
 *   foreground: `hsl(score * 120, var(--score-fg-s), var(--score-fg-l))`
 *   background: RGB-interpolated, alpha scaled by `var(--score-bg-a-mul)`
 *
 * Saturation, lightness, and bg alpha are driven by per-theme CSS vars so
 * yellow mid-tones stay legible on the warm-cream light canvas without
 * breaking the dark theme's established appearance. See index.css
 * `--score-fg-s` / `--score-fg-l` / `--score-bg-a-mul`.
 *
 * These helpers back `<ScoreChip>`, `<ScoreBadge>`, `<ScoreRing>`, dataset
 * chips and the group-header best-score callouts. Anything that paints a
 * dynamic score MUST funnel through here — do not roll a parallel palette.
 */

/** HSL foreground: `hsl(hue var(--score-fg-s) var(--score-fg-l))` clamped to [0, 1]. */
export function scoreColor(score: number): string {
  const hue = Math.round(Math.max(0, Math.min(1, score)) * 120)
  return `hsl(${hue} var(--score-fg-s, 70%) var(--score-fg-l, 45%))`
}

/**
 * Translucent score background. RGB-interpolated for richer mid-tones than a
 * single HSL alpha (kept for visual continuity with existing chips/badges).
 * Final alpha is `alpha * var(--score-bg-a-mul, 1)` — light theme boosts the
 * multiplier so washed-out yellows actually paint a visible pill on cream.
 */
export function scoreBg(score: number, alpha = 0.35): string {
  const t = Math.max(0, Math.min(1, score))
  let r: number, g: number, b: number

  if (t < 0.5) {
    const s = t / 0.5
    r = 215 + (255 - 215) * s
    g = 48 + (255 - 48) * s
    b = 39 + (0 - 39) * s
  } else {
    const s = (t - 0.5) / 0.5
    r = 255 + (26 - 255) * s
    g = 255 + (150 - 255) * s
    b = 0 + (65 - 0) * s
  }
  return `rgb(${Math.round(r)} ${Math.round(g)} ${Math.round(b)} / calc(${alpha} * var(--score-bg-a-mul, 1)))`
}

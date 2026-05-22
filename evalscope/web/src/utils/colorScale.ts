/**
 * Score → color helpers.
 * The product's emotional core — see DESIGN.md §Score Gradient.
 *
 *   foreground: `hsl(score * 120, 70%, 45%)`  (0 → red, 0.5 → yellow, 1 → green)
 *   background: same hue, low alpha
 *
 * These helpers back `<ScoreChip>`, `<ScoreBadge>`, dataset chips and the
 * group-header best-score callouts. Anything that paints a dynamic score MUST
 * funnel through here — do not roll a parallel palette.
 */

/** HSL foreground: `hsl(score * 120, 70%, 45%)` clamped to [0, 1]. */
export function scoreColor(score: number): string {
  const hue = Math.round(Math.max(0, Math.min(1, score)) * 120)
  return `hsl(${hue}, 70%, 45%)`
}

/**
 * Translucent score background. RGB-interpolated for richer mid-tones than a
 * single HSL alpha (kept for visual continuity with existing chips/badges).
 * Default alpha 0.35 matches the existing chip/badge appearance.
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
  return `rgba(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)}, ${alpha})`
}

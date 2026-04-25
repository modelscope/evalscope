/**
 * RdYlGn color scale: maps a score [0, 1] to an RGB color.
 * Red (0) → Yellow (0.5) → Green (1)
 */
export function rdYlGn(score: number): string {
  const t = Math.max(0, Math.min(1, score))
  let r: number, g: number, b: number

  if (t < 0.5) {
    // Red → Yellow
    const s = t / 0.5
    r = 215 + (255 - 215) * s
    g = 48 + (255 - 48) * s
    b = 39 + (0 - 39) * s
  } else {
    // Yellow → Green
    const s = (t - 0.5) / 0.5
    r = 255 + (26 - 255) * s
    g = 255 + (150 - 255) * s
    b = 0 + (65 - 0) * s
  }
  return `rgb(${Math.round(r)}, ${Math.round(g)}, ${Math.round(b)})`
}

/** Score cell background color at reduced opacity for table use. */
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

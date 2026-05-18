/**
 * Format milliseconds: displays as "Xms" when < 1000, otherwise "X.Xs".
 * Returns empty string for null/undefined.
 */
export function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return ''
  if (ms < 1000) return `${ms.toFixed(0)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

/**
 * Format seconds value to "X.XXs". Returns null for null/undefined.
 */
export function fmtSec(n: number | null | undefined): string | null {
  if (n == null) return null
  return `${n.toFixed(2)}s`
}

export function formatScore(score: number, precision = 4): string {
  return score.toFixed(precision)
}

export function formatPercent(value: number): string {
  return `${(value * 100).toFixed(1)}%`
}

export function prettyJson(value: unknown): string {
  try {
    return JSON.stringify(value, null, 2)
  } catch {
    return String(value)
  }
}

/**
 * Format milliseconds: displays as "Xms" when < 1000, otherwise "X.Xs".
 * Returns empty string for null/undefined.
 */
export function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return ''
  if (ms < 1000) return `${ms.toFixed(0)}ms`
  return `${(ms / 1000).toFixed(1)}s`
}

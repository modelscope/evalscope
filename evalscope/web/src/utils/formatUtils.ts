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

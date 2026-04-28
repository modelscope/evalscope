export function normalizeScore(score: unknown): number {
  if (typeof score === 'boolean') return score ? 1.0 : 0.0
  if (typeof score === 'number') return score
  if (typeof score === 'object' && score !== null) {
    const vals = Object.values(score as Record<string, unknown>)
    if (vals.length > 0) {
      const n = Number(vals[0])
      return isNaN(n) ? 0.0 : n
    }
    return 0.0
  }
  const n = Number(score)
  return isNaN(n) ? 0.0 : n
}

export function isPass(nscore: number, threshold: number): boolean {
  return nscore >= threshold
}
